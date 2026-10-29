/**
 * @file convolution.cu
 * @brief PyTorch C++/CUDA extension dispatch binding for the seven convolution algorithms.
 *
 * Implements input validation, memory allocation, launch configuration calculation,
 * and dispatch logic for seven custom CUDA convolution variants.
 */

#include <torch/extension.h>
#include <map>
#include <string>
#include <algorithm>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "../convolution/cuda/conv2d_kernels.cuh"

/**
 * @brief Retrieve and cache hardware device properties for the specified GPU device.
 *
 * Caches device properties outside the measured hot path after initial invocation.
 * Python API callers retain the GIL, serializing initialization at the host API boundary.
 *
 * @param device CUDA device ordinal.
 * @return const cudaDeviceProp& Cached device properties reference.
 */
const cudaDeviceProp& device_properties(int device) {
    static std::map<int, cudaDeviceProp> cache;
    auto it = cache.find(device);
    if (it == cache.end()) {
        cudaDeviceProp prop;
        C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        it = cache.emplace(device, prop).first;
    }
    return it->second;
}

/**
 * @brief Dispatch a 2D cross-correlation convolution to the chosen CUDA kernel variant.
 *
 * Validates tensor constraints (contiguous FP32 CUDA tensors), derives spatial output
 * dimensions P and Q, allocates the output tensor, and schedules the corresponding CUDA kernel.
 *
 * @param x Input tensor [N, C, H, W] in device memory.
 * @param w Filter weight tensor [K, C, R, S] in device memory.
 * @param b 1D bias vector [K] in device memory.
 * @param method Algorithm identifier ('naive', 'const', 'tiled', 'im2col', 'im2col_reg', 'implicit', 'gridstride').
 * @param stride Spatial stride step along height and width (stride > 0).
 * @param padding Symmetric zero-padding along height and width (padding >= 0).
 * @param tile Block tile dimension for shared-memory algorithms (8, 16, or 32).
 * @param grid_blocks Number of launched thread blocks for gridstride (0 = 8 times the SM count).
 * @return torch::Tensor Output convolved tensor [N, K, P, Q] on the same device.
 */
torch::Tensor convolution(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    std::string method,
    int stride,
    int padding,
    int tile,
    int grid_blocks
) {
    // 1. Validation checks
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
    TORCH_CHECK(stride > 0 && padding >= 0 && tile > 0 && grid_blocks >= 0, "Invalid launch configuration parameters");
    TORCH_CHECK(
        method == "naive" || method == "const" || method == "tiled" ||
        method == "im2col" || method == "im2col_reg" || method == "implicit" ||
        method == "gridstride",
        "Unknown convolution method: " + method
    );

    // 2. Geometry configuration
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

    dim3 block(16, 16);
    dim3 grid((d.Q + 15) / 16, (d.P + 15) / 16, std::min(d.N * d.K, 65535));

    // 3. Kernel Dispatch
    if (method == "naive") {
        conv2d_naive<<<grid, block, 0, stream>>>(d, xp, wp, bp, yp);
    } else if (method == "const") {
        TORCH_CHECK(w.numel() <= CONV_MAX_CONST_W, "N/A: weight exceeds 64 KiB constant memory");
        TORCH_CHECK(
            stream == c10::cuda::getDefaultCUDAStream(x.get_device()),
            "constant memory path requires default CUDA stream"
        );
        C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(
            c_weight, wp, w.numel() * sizeof(float), 0, cudaMemcpyDeviceToDevice, stream
        ));
        conv2d_const<<<grid, block, 0, stream>>>(d, xp, bp, yp);
    } else if (method == "tiled") {
        TORCH_CHECK(tile == 8 || tile == 16 || tile == 32, "tile must be 8, 16, or 32");
        size_t bytes = tiled_smem_bytes(tile, d);
        const auto& prop = device_properties(x.get_device());
        TORCH_CHECK(bytes <= prop.sharedMemPerBlock, "N/A: tile exceeds shared memory capacity");
        dim3 tb(tile, tile);
        dim3 tg((d.Q + tile - 1) / tile, (d.P + tile - 1) / tile, std::min(d.N * d.K, 65535));
        if (tile == 8) {
            conv2d_tiled<8><<<tg, tb, bytes, stream>>>(d, xp, wp, bp, yp);
        } else if (tile == 16) {
            conv2d_tiled<16><<<tg, tb, bytes, stream>>>(d, xp, wp, bp, yp);
        } else {
            conv2d_tiled<32><<<tg, tb, bytes, stream>>>(d, xp, wp, bp, yp);
        }
    } else if (method == "im2col" || method == "im2col_reg") {
        const int kd = d.C * d.R * d.S;
        const int pq = d.P * d.Q;
        auto col = torch::empty({kd, pq}, x.options());
        for (int n = 0; n < d.N; ++n) {
            int blocks = static_cast<int>(std::min((col.numel() + 255) / 256, static_cast<int64_t>(65535)));
            im2col_kernel<<<blocks, 256, 0, stream>>>(d, xp, n, 1, col.data_ptr<float>());
            ConvEpilogue ep{yp, bp, d.K, pq, n};
            if (method == "im2col") {
                gemm_tiled<16><<<dim3((pq + 15) / 16, (d.K + 15) / 16), dim3(16, 16), 0, stream>>>(
                    d.K, pq, kd, wp, col.data_ptr<float>(), ep
                );
            } else if (d.K >= 32) {
                gemm_reg<64, 64, 8, 4, 4><<<dim3((pq + 63) / 64, (d.K + 63) / 64), 256, 0, stream>>>(
                    d.K, pq, kd, wp, col.data_ptr<float>(), ep
                );
            } else {
                gemm_reg<16, 128, 16, 2, 8><<<dim3((pq + 127) / 128, (d.K + 15) / 16), 128, 0, stream>>>(
                    d.K, pq, kd, wp, col.data_ptr<float>(), ep
                );
            }
        }
    } else if (method == "implicit") {
        const auto& prop = device_properties(x.get_device());
        if (implicit_use_reg(d, prop.sharedMemPerBlock, prop.multiProcessorCount)) {
            launch_implicit_reg_for_k(d, xp, wp, bp, yp, stream);
        } else {
            int64_t cols = static_cast<int64_t>(d.N) * d.P * d.Q;
            conv2d_implicit_gemm<16><<<dim3((cols + 15) / 16, (d.K + 15) / 16), dim3(16, 16), 0, stream>>>(
                d, xp, wp, bp, yp
            );
        }
    } else {
        const auto& prop = device_properties(x.get_device());
        int blocks = grid_blocks ? grid_blocks : prop.multiProcessorCount * 8;
        conv2d_gridstride<<<blocks, 256, 0, stream>>>(d, xp, wp, bp, yp, nullptr);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convolution", &convolution, "Dispatch 2D convolution across seven custom CUDA kernel variants");
}
