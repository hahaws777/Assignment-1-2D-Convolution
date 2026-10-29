/**
 * @file implicit_reg.cu
 * @brief Extension experiment: register-blocked implicit GEMM convolution.
 *
 * conv2d_implicit_reg now lives in conv2d_kernels.cuh and is what the published "implicit"
 * method runs in auto mode. This binding keeps explicit mode control for the extension
 * experiments, so they can still compare against the original conv2d_implicit_gemm (mode 2).
 * Compared with conv2d_implicit_gemm, conv2d_implicit_reg:
 *   1. accumulates a TM x TN output tile per thread in registers, like gemm_reg;
 *   2. picks the block tile height from K;
 *   3. decodes (c, r, s) and each output column once per block into shared-memory tables.
 * It still gathers input directly (no im2col buffer) and fuses the bias, like implicit.
 */

#include <torch/extension.h>
#include <algorithm>
#include <climits>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "../convolution/cuda/conv2d_kernels.cuh"

/**
 * @brief Register-blocked implicit GEMM convolution with the same validation as convolution().
 *
 * @param mode 0 = auto (same selection as the published "implicit" method: original implicit
 *             when the register tiles would launch fewer blocks than the GPU has SMs, or the
 *             reduction table does not fit), 1 = always register tiles, 2 = always the original
 *             conv2d_implicit_gemm<16>.
 */
torch::Tensor implicit_reg(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    int stride,
    int padding,
    int mode
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
    TORCH_CHECK(stride > 0 && padding >= 0 && mode >= 0 && mode <= 2, "Invalid launch configuration parameters");
    TORCH_CHECK(x.size(1) * x.size(2) * x.size(3) <= INT_MAX, "Per-image input exceeds 32-bit offsets");

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

    const auto* prop = at::cuda::getDeviceProperties(x.get_device());
    TORCH_CHECK(mode != 1 || implicit_reg_table_fits(d, prop->sharedMemPerBlock),
                "N/A: reduction table exceeds shared memory capacity");
    const bool use_reg = mode == 1 ||
        (mode == 0 && implicit_use_reg(d, prop->sharedMemPerBlock, prop->multiProcessorCount));

    if (use_reg) {
        launch_implicit_reg_for_k(d, xp, wp, bp, yp, stream);
    } else {
        const long long cols = (long long)d.N * d.P * d.Q;
        conv2d_implicit_gemm<16><<<dim3((cols + 15) / 16, (d.K + 15) / 16), dim3(16, 16), 0, stream>>>(
            d, xp, wp, bp, yp
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("implicit_reg", &implicit_reg, "Register-blocked implicit GEMM convolution (mode 0 auto, 1 reg, 2 original)");
}
