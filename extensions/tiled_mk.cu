/**
 * @file tiled_mk.cu
 * @brief Extension experiment: shared-memory tiling that computes KB output channels per thread.
 *
 * Not part of the published seven variants. The original conv2d_tiled reloads the same
 * input tile once per output channel and performs only R*S multiply-adds between barriers.
 * conv2d_tiled_mk stages the tile once per group of KB filters and reuses each staged input
 * value KB times from a register. KB = 1 reproduces the original loop structure.
 */

#include <torch/extension.h>
#include <algorithm>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "../convolution/cuda/conv2d_kernels.cuh"

/**
 * @brief Shared-memory tiled convolution computing KB output channels per thread.
 *
 * Shared memory holds one input tile plus halo and KB filter slices for the current
 * input channel. Accumulators live in registers; bias is added after summation, matching
 * the order used by conv2d_tiled.
 *
 * @tparam T Block dimension along spatial X and Y (8, 16, or 32).
 * @tparam KB Output channels computed per thread.
 */
template <int T, int KB>
__global__ void __launch_bounds__(T * T) conv2d_tiled_mk(
    ConvDims d,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y
) {
    extern __shared__ float smem[];
    const int tileH = tiled_tile_h(T, d.stride, d.R);
    const int tileW = tiled_tile_w(T, d.stride, d.S);
    const int RS = d.R * d.S;
    float* s_in = smem;
    float* s_w = smem + tileH * tileW;  // KB filters x R x S

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * T + tx;
    const int q0 = blockIdx.x * T;
    const int p0 = blockIdx.y * T;
    const int q = q0 + tx;
    const int p = p0 + ty;
    const int ih_base = p0 * d.stride - d.pad;
    const int iw_base = q0 * d.stride - d.pad;
    const bool active = (p < d.P) && (q < d.Q);
    const int groups = (d.K + KB - 1) / KB;

    for (int ng = blockIdx.z; ng < d.N * groups; ng += gridDim.z) {
        const int n = ng / groups;
        const int k0 = (ng % groups) * KB;
        float acc[KB];
#pragma unroll
        for (int j = 0; j < KB; ++j) {
            acc[j] = 0.f;
        }

        for (int c = 0; c < d.C; ++c) {
            // Cooperative load of input tile + halo (once per filter group, not per filter)
            const float* xplane = x + idx4(n, c, 0, 0, d.C, d.H, d.W);
            for (int i = tid; i < tileH * tileW; i += T * T) {
                const int ly = i / tileW;
                const int lx = i % tileW;
                const int ih = ih_base + ly;
                const int iw = iw_base + lx;
                s_in[i] = (ih >= 0 && ih < d.H && iw >= 0 && iw < d.W) ? xplane[(size_t)ih * d.W + iw] : 0.f;
            }

            // Cooperative load of KB filter slices; filters past K are zero-filled
            for (int i = tid; i < KB * RS; i += T * T) {
                const int j = i / RS;
                const int rs = i % RS;
                s_w[i] = (k0 + j < d.K) ? w[idx4(k0 + j, c, 0, 0, d.C, d.R, d.S) + rs] : 0.f;
            }
            __syncthreads();

            // Each staged input value is read once and reused for KB filters
            if (active) {
                const float* sp = s_in + (ty * d.stride) * tileW + tx * d.stride;
                for (int r = 0; r < d.R; ++r) {
                    for (int s = 0; s < d.S; ++s) {
                        const float v = sp[r * tileW + s];
                        const float* wp = s_w + r * d.S + s;
#pragma unroll
                        for (int j = 0; j < KB; ++j) {
                            acc[j] += v * wp[j * RS];
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (active) {
#pragma unroll
            for (int j = 0; j < KB; ++j) {
                if (k0 + j < d.K) {
                    y[idx4(n, k0 + j, p, q, d.K, d.P, d.Q)] = acc[j] + (d.has_bias ? b[k0 + j] : 0.f);
                }
            }
        }
    }
}

/**
 * @brief Size shared memory, check capacity, and launch conv2d_tiled_mk<T, KB>.
 */
template <int T, int KB>
void launch_tiled_mk(const ConvDims& d, const float* xp, const float* wp, const float* bp, float* yp,
                     int device, cudaStream_t stream) {
    const size_t bytes = sizeof(float) * (
        (size_t)tiled_tile_h(T, d.stride, d.R) * tiled_tile_w(T, d.stride, d.S) +
        (size_t)KB * d.R * d.S
    );
    TORCH_CHECK(bytes <= at::cuda::getDeviceProperties(device)->sharedMemPerBlock,
                "N/A: tile exceeds shared memory capacity");
    const int groups = (d.K + KB - 1) / KB;
    dim3 tb(T, T);
    dim3 tg((d.Q + T - 1) / T, (d.P + T - 1) / T, std::min(d.N * groups, 65535));
    conv2d_tiled_mk<T, KB><<<tg, tb, bytes, stream>>>(d, xp, wp, bp, yp);
}

/**
 * @brief Select the KB instantiation for a fixed tile size.
 */
template <int T>
void dispatch_kb(int kb, const ConvDims& d, const float* xp, const float* wp, const float* bp, float* yp,
                 int device, cudaStream_t stream) {
    switch (kb) {
        case 1: launch_tiled_mk<T, 1>(d, xp, wp, bp, yp, device, stream); break;
        case 4: launch_tiled_mk<T, 4>(d, xp, wp, bp, yp, device, stream); break;
        case 8: launch_tiled_mk<T, 8>(d, xp, wp, bp, yp, device, stream); break;
        case 16: launch_tiled_mk<T, 16>(d, xp, wp, bp, yp, device, stream); break;
        default: TORCH_CHECK(false, "kb must be 1, 4, 8, or 16");
    }
}

/**
 * @brief Multi-output-channel tiled convolution with the same validation as convolution().
 *
 * @param kb Output channels per thread (1, 4, 8, or 16).
 */
torch::Tensor tiled_mk(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    int stride,
    int padding,
    int tile,
    int kb
) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && b.is_cuda(), "CUDA required for all operands");
    TORCH_CHECK(x.device() == w.device() && x.device() == b.device(), "Tensors must reside on the same CUDA device");
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat32 &&
        w.scalar_type() == torch::kFloat32 &&
        b.scalar_type() == torch::kFloat32,
        "FP32 tensors required"
    );
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && b.is_contiguous(), "All tensors must be contiguous");
    TORCH_CHECK(x.dim() == 4 && w.dim() == 4 && b.dim() == 1 && x.numel() && w.numel(), "Invalid tensor dimensions");
    TORCH_CHECK(x.size(1) == w.size(1) && b.size(0) == w.size(0), "Channel dimensions mismatch");
    TORCH_CHECK(stride > 0 && padding >= 0, "Invalid launch configuration parameters");

    ConvDims d{
        static_cast<int>(x.size(0)), static_cast<int>(x.size(1)),
        static_cast<int>(x.size(2)), static_cast<int>(x.size(3)),
        static_cast<int>(w.size(0)), static_cast<int>(w.size(2)),
        static_cast<int>(w.size(3)), stride, padding, 0, 0, 1
    };

    TORCH_CHECK(d.H + 2 * padding >= d.R && d.W + 2 * padding >= d.S, "Kernel size exceeds padded input dimensions");
    d.P = (d.H + 2 * padding - d.R) / stride + 1;
    d.Q = (d.W + 2 * padding - d.S) / stride + 1;

    c10::cuda::CUDAGuard guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto y = torch::empty({d.N, d.K, d.P, d.Q}, x.options());

    const float* xp = x.data_ptr<float>();
    const float* wp = w.data_ptr<float>();
    const float* bp = b.data_ptr<float>();
    float* yp = y.data_ptr<float>();
    const int device = x.get_device();

    if (tile == 8) {
        dispatch_kb<8>(kb, d, xp, wp, bp, yp, device, stream);
    } else if (tile == 16) {
        dispatch_kb<16>(kb, d, xp, wp, bp, yp, device, stream);
    } else if (tile == 32) {
        dispatch_kb<32>(kb, d, xp, wp, bp, yp, device, stream);
    } else {
        TORCH_CHECK(false, "tile must be 8, 16, or 32");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiled_mk", &tiled_mk, "Shared-memory tiled convolution with KB output channels per thread");
}
