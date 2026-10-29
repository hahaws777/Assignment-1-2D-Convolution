/**
 * @file conv2d_kernels.cuh
 * @brief CUDA kernel implementations for 2D convolution (NCHW, FP32, cross-correlation).
 *
 * Implements seven algorithmic convolution variants exploring memory reuse and scheduling:
 *  1. conv2d_naive: One thread per output pixel, direct global memory access.
 *  2. conv2d_const: Filter weights stored in __constant__ memory for warp-wide broadcasts.
 *  3. conv2d_tiled: Cooperative shared-memory staging of input tile and halo.
 *  4. im2col + gemm_tiled: Explicit matrix lowering and shared-memory tiled GEMM.
 *  5. im2col + gemm_reg: Matrix lowering with 2D register-blocked GEMM.
 *  6. implicit: Fused GEMM gathering input elements on-the-fly without scratchpad. Runs the
 *     register-tiled conv2d_implicit_reg, falling back to conv2d_implicit_gemm when
 *     implicit_use_reg says the register tiles would underfill the GPU.
 *  7. conv2d_gridstride: Grid-stride direct convolution with a configurable thread-block count.
 *
 * Implemented natively in CUDA without cuBLAS or cuDNN.
 */

#pragma once
#include <climits>
#include <cuda_runtime.h>

/**
 * @struct ConvDims
 * @brief Geometric dimensions and parameters for 2D convolution.
 *
 * Passed by value to kernels and stored in constant parameter cache.
 */
struct ConvDims {
    int N;          /**< Batch size (number of images) */
    int C;          /**< Input feature map channel depth */
    int H;          /**< Input feature map height */
    int W;          /**< Input feature map width */
    int K;          /**< Output channel count (number of filters) */
    int R;          /**< Filter kernel spatial height */
    int S;          /**< Filter kernel spatial width */
    int stride;     /**< Convolution stride along H and W */
    int pad;        /**< Symmetric zero-padding along H and W */
    int P;          /**< Output feature map height: (H + 2*pad - R) / stride + 1 */
    int Q;          /**< Output feature map width:  (W + 2*pad - S) / stride + 1 */
    int has_bias;   /**< Non-zero if bias vector is applied */
};

/**
 * @def CONV_MAX_CONST_W
 * @brief Maximum capacity of device constant memory for filter weights (64 KiB = 16384 FP32 floats).
 */
#define CONV_MAX_CONST_W 16384

/**
 * @brief Constant memory symbol storing weight tensors for conv2d_const.
 */
__constant__ float c_weight[CONV_MAX_CONST_W];

/**
 * @brief Compute flat 1D memory offset for a 4D tensor in row-major layout.
 *
 * @param a Dimension index 0 (e.g., n)
 * @param b Dimension index 1 (e.g., c)
 * @param c Dimension index 2 (e.g., h)
 * @param d Dimension index 3 (e.g., w)
 * @param B Extent of dimension 1
 * @param C Extent of dimension 2
 * @param D Extent of dimension 3
 * @return size_t Flat offset into contiguous linear memory.
 */
__device__ __forceinline__ size_t idx4(int a, int b, int c, int d, int B, int C, int D) {
    return (((size_t)a * B + b) * C + c) * D + d;
}

// ---------------------------------------------------------------------------
// 1. Naive Direct Convolution
// ---------------------------------------------------------------------------

/**
 * @brief Naive direct 2D convolution kernel.
 *
 * Maps one thread to each output pixel (q, p). Reads input pixels and filter weights
 * directly from global memory without explicit staging.
 *
 * @param d Convolution dimensional metadata.
 * @param x Input tensor [N, C, H, W] in global device memory.
 * @param w Filter weights [K, C, R, S] in global device memory.
 * @param b Optional bias vector [K] in global device memory.
 * @param y Output tensor [N, K, P, Q] in global device memory.
 */
__global__ void conv2d_naive(
    ConvDims d,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    const int p = blockIdx.y * blockDim.y + threadIdx.y;
    if (p >= d.P || q >= d.Q) return;

    for (int nk = blockIdx.z; nk < d.N * d.K; nk += gridDim.z) {
        const int n = nk / d.K;
        const int k = nk % d.K;
        float acc = d.has_bias ? b[k] : 0.f;

        for (int c = 0; c < d.C; ++c) {
            for (int r = 0; r < d.R; ++r) {
                const int ih = p * d.stride - d.pad + r;
                if (ih < 0 || ih >= d.H) continue;
                for (int s = 0; s < d.S; ++s) {
                    const int iw = q * d.stride - d.pad + s;
                    if (iw < 0 || iw >= d.W) continue;
                    acc += x[idx4(n, c, ih, iw, d.C, d.H, d.W)] * w[idx4(k, c, r, s, d.C, d.R, d.S)];
                }
            }
        }
        y[idx4(n, k, p, q, d.K, d.P, d.Q)] = acc;
    }
}

// ---------------------------------------------------------------------------
// 2. Constant-Memory Direct Convolution
// ---------------------------------------------------------------------------

/**
 * @brief Direct 2D convolution reading weights from GPU constant cache (__constant__).
 *
 * Filter weights reside in constant memory (`c_weight`). Threads within a warp
 * access the same filter weight simultaneously, achieving uniform broadcast reads.
 *
 * @param d Convolution dimensional metadata.
 * @param x Input tensor [N, C, H, W] in global memory.
 * @param b Bias vector [K] in global memory.
 * @param y Output tensor [N, K, P, Q] in global memory.
 */
__global__ void conv2d_const(
    ConvDims d,
    const float* __restrict__ x,
    const float* __restrict__ b,
    float* __restrict__ y
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    const int p = blockIdx.y * blockDim.y + threadIdx.y;
    if (p >= d.P || q >= d.Q) return;

    for (int nk = blockIdx.z; nk < d.N * d.K; nk += gridDim.z) {
        const int n = nk / d.K;
        const int k = nk % d.K;
        float acc = d.has_bias ? b[k] : 0.f;

        for (int c = 0; c < d.C; ++c) {
            for (int r = 0; r < d.R; ++r) {
                const int ih = p * d.stride - d.pad + r;
                if (ih < 0 || ih >= d.H) continue;
                const float* xrow = x + idx4(n, c, ih, 0, d.C, d.H, d.W);
                const float* wrow = c_weight + idx4(k, c, r, 0, d.C, d.R, d.S);
                for (int s = 0; s < d.S; ++s) {
                    const int iw = q * d.stride - d.pad + s;
                    if (iw < 0 || iw >= d.W) continue;
                    acc += xrow[iw] * wrow[s];
                }
            }
        }
        y[idx4(n, k, p, q, d.K, d.P, d.Q)] = acc;
    }
}

// ---------------------------------------------------------------------------
// 3. Shared-Memory Tiling with Halo
// ---------------------------------------------------------------------------

/**
 * @brief Calculate vertical dimension of shared-memory input tile including halo.
 */
__host__ __device__ inline int tiled_tile_h(int T, int stride, int R) {
    return (T - 1) * stride + R;
}

/**
 * @brief Calculate horizontal dimension of shared-memory input tile including halo.
 */
__host__ __device__ inline int tiled_tile_w(int T, int stride, int S) {
    return (T - 1) * stride + S;
}

/**
 * @brief Calculate required dynamic shared memory bytes for conv2d_tiled.
 */
inline size_t tiled_smem_bytes(int T, const ConvDims& d) {
    return sizeof(float) * (
        (size_t)tiled_tile_h(T, d.stride, d.R) * tiled_tile_w(T, d.stride, d.S) +
        (size_t)d.R * d.S
    );
}

/**
 * @brief 2D Convolution with shared-memory input tile and halo staging.
 *
 * A thread block of size T x T cooperatively loads an input tile plus halo into
 * shared memory, reusing each input pixel across multiple filter taps.
 *
 * @tparam T Block dimension along spatial X and Y (8, 16, or 32).
 * @param d Convolution dimensional metadata.
 * @param x Input tensor [N, C, H, W] in global memory.
 * @param w Filter weights [K, C, R, S] in global memory.
 * @param b Bias vector [K] in global memory.
 * @param y Output tensor [N, K, P, Q] in global memory.
 */
template <int T>
__global__ void __launch_bounds__(T * T) conv2d_tiled(
    ConvDims d,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y
) {
    extern __shared__ float smem[];
    const int tileH = tiled_tile_h(T, d.stride, d.R);
    const int tileW = tiled_tile_w(T, d.stride, d.S);
    float* s_in = smem;
    float* s_w = smem + tileH * tileW;

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

    for (int nk = blockIdx.z; nk < d.N * d.K; nk += gridDim.z) {
        const int n = nk / d.K;
        const int k = nk % d.K;
        float acc = 0.f;

        for (int c = 0; c < d.C; ++c) {
            // Cooperative load of input tile + halo
            const float* xplane = x + idx4(n, c, 0, 0, d.C, d.H, d.W);
            for (int i = tid; i < tileH * tileW; i += T * T) {
                const int ly = i / tileW;
                const int lx = i % tileW;
                const int ih = ih_base + ly;
                const int iw = iw_base + lx;
                s_in[i] = (ih >= 0 && ih < d.H && iw >= 0 && iw < d.W) ? xplane[(size_t)ih * d.W + iw] : 0.f;
            }

            // Cooperative load of filter slice
            const float* wslice = w + idx4(k, c, 0, 0, d.C, d.R, d.S);
            for (int i = tid; i < d.R * d.S; i += T * T) {
                s_w[i] = wslice[i];
            }
            __syncthreads();

            // Compute accumulation from shared memory
            if (active) {
                const float* sp = s_in + (ty * d.stride) * tileW + tx * d.stride;
                for (int r = 0; r < d.R; ++r) {
                    for (int s = 0; s < d.S; ++s) {
                        acc += sp[r * tileW + s] * s_w[r * d.S + s];
                    }
                }
            }
            __syncthreads();
        }

        if (active) {
            y[idx4(n, k, p, q, d.K, d.P, d.Q)] = acc + (d.has_bias ? b[k] : 0.f);
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Im2Col Lowering + GEMM
// ---------------------------------------------------------------------------

/**
 * @brief Explicit im2col lowering kernel.
 *
 * Flattens overlapping input image receptive fields into a column matrix
 * of shape (C*R*S) x (nb*P*Q) for subsequent matrix multiplication.
 *
 * @param d Convolution dimensional metadata.
 * @param x Input tensor [N, C, H, W] in device memory.
 * @param n0 Starting image batch index.
 * @param nb Number of images lowered in this chunk.
 * @param col Destination output matrix buffer in device memory.
 */
__global__ void im2col_kernel(
    ConvDims d,
    const float* __restrict__ x,
    int n0,
    int nb,
    float* __restrict__ col
) {
    const size_t PQ = (size_t)d.P * d.Q;
    const size_t ncol = (size_t)nb * PQ;
    const size_t total = (size_t)d.C * d.R * d.S * ncol;

    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t j = idx % ncol;
        const int crs = (int)(idx / ncol);
        const int s = crs % d.S;
        const int r = (crs / d.S) % d.R;
        const int c = crs / (d.R * d.S);
        const int nl = (int)(j / PQ);
        const int pq = (int)(j % PQ);
        const int p = pq / d.Q;
        const int q = pq % d.Q;
        const int ih = p * d.stride - d.pad + r;
        const int iw = q * d.stride - d.pad + s;
        col[idx] = (ih >= 0 && ih < d.H && iw >= 0 && iw < d.W)
                       ? x[idx4(n0 + nl, c, ih, iw, d.C, d.H, d.W)]
                       : 0.f;
    }
}

/**
 * @struct ConvEpilogue
 * @brief Helper for writing GEMM results directly into NCHW output layout with bias.
 */
struct ConvEpilogue {
    float* y;
    const float* bias;
    int K, PQ, n0;

    __device__ __forceinline__ void store(int row, int col, float v) const {
        const int nl = col / PQ;
        const int pq = col % PQ;
        if (bias) v += bias[row];
        y[((size_t)(n0 + nl) * K + row) * PQ + pq] = v;
    }
};

/**
 * @brief Shared-memory 2D tiled GEMM: C[M x Nc] = A[M x Kd] * B[Kd x Nc].
 *
 * @tparam T Tile dimension (e.g. 16).
 * @param M Row dimension of matrix A and C (number of output filters K).
 * @param Nc Column dimension of matrix B and C (P * Q).
 * @param Kd Inner reduction dimension (C * R * S).
 * @param A Filter weight matrix.
 * @param B Lowered im2col matrix.
 * @param e Output epilogue storing convolved result.
 */
template <int T>
__global__ void __launch_bounds__(T * T) gemm_tiled(
    int M,
    int Nc,
    int Kd,
    const float* __restrict__ A,
    const float* __restrict__ B,
    ConvEpilogue e
) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * T + ty;
    const int col = blockIdx.x * T + tx;
    float acc = 0.f;

    for (int t = 0; t < Kd; t += T) {
        As[ty][tx] = (row < M && t + tx < Kd) ? A[(size_t)row * Kd + t + tx] : 0.f;
        Bs[ty][tx] = (t + ty < Kd && col < Nc) ? B[(size_t)(t + ty) * Nc + col] : 0.f;
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < T; ++kk) {
            acc += As[ty][kk] * Bs[kk][tx];
        }
        __syncthreads();
    }

    if (row < M && col < Nc) {
        e.store(row, col, acc);
    }
}

// ---------------------------------------------------------------------------
// 5. Register-Blocked GEMM
// ---------------------------------------------------------------------------

/**
 * @brief High-performance GEMM with 2D register blocking and bank conflict avoidance.
 *
 * Each thread maintains a TM x TN accumulator tile in registers, amortizing shared
 * memory read traffic over TM*TN arithmetic multiply-accumulate operations.
 *
 * @tparam BM Block tile height along M dimension.
 * @tparam BN Block tile width along N dimension.
 * @tparam BK K reduction dimension chunk size.
 * @tparam TM Number of M elements computed per thread in registers.
 * @tparam TN Number of N elements computed per thread in registers.
 */
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN)) gemm_reg(
    int M,
    int Nc,
    int Kd,
    const float* __restrict__ A,
    const float* __restrict__ B,
    ConvEpilogue e
) {
    constexpr int NT = (BM / TM) * (BN / TN);
    __shared__ float As[BK][BM + 4];  // +4 padding prevents bank conflicts
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int tRow = tid / (BN / TN);
    const int tCol = tid % (BN / TN);
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.f;
        }
    }
    float ra[TM], rb[TN];

    for (int t = 0; t < Kd; t += BK) {
        for (int i = tid; i < BM * BK; i += NT) {
            const int m = i / BK;
            const int kk = i % BK;
            const int gr = rowBase + m;
            const int gk = t + kk;
            As[kk][m] = (gr < M && gk < Kd) ? A[(size_t)gr * Kd + gk] : 0.f;
        }
        for (int i = tid; i < BK * BN; i += NT) {
            const int kk = i / BN;
            const int nn = i % BN;
            const int gk = t + kk;
            const int gc = colBase + nn;
            Bs[kk][nn] = (gk < Kd && gc < Nc) ? B[(size_t)gk * Nc + gc] : 0.f;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                ra[i] = As[kk][tRow * TM + i];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                rb[j] = Bs[kk][tCol * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += ra[i] * rb[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int r = rowBase + tRow * TM + i;
            const int c = colBase + tCol * TN + j;
            if (r < M && c < Nc) {
                e.store(r, c, acc[i][j]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Implicit GEMM Convolution
// ---------------------------------------------------------------------------

/**
 * @brief Implicit GEMM convolution kernel gathering matrix tiles on-the-fly.
 *
 * Evaluates convolution as a GEMM without materializing the im2col scratchpad buffer.
 * Tile elements of operand B are decoded directly into input tensor coordinates.
 *
 * @tparam T Tile spatial dimension (e.g. 16).
 * @param d Convolution dimensional metadata.
 * @param x Input tensor [N, C, H, W] in device memory.
 * @param w Filter weights [K, C, R, S] in device memory.
 * @param b Bias vector [K] in device memory.
 * @param y Output tensor [N, K, P, Q] in device memory.
 */
template <int T>
__global__ void __launch_bounds__(T * T) conv2d_implicit_gemm(
    ConvDims d,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y
) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    const int M = d.K;
    const int Kd = d.C * d.R * d.S;
    const int PQ = d.P * d.Q;
    const long long Nc = (long long)d.N * PQ;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * T + ty;
    const long long colA = (long long)blockIdx.x * T + tx;

    const bool colOk = colA < Nc;
    const int n = colOk ? (int)(colA / PQ) : 0;
    const int pq = colOk ? (int)(colA % PQ) : 0;
    const int p = pq / d.Q;
    const int q = pq % d.Q;
    const int ihb = p * d.stride - d.pad;
    const int iwb = q * d.stride - d.pad;
    float acc = 0.f;

    for (int t = 0; t < Kd; t += T) {
        As[ty][tx] = (row < M && t + tx < Kd) ? w[(size_t)row * Kd + t + tx] : 0.f;
        const int gk = t + ty;
        float v = 0.f;

        if (colOk && gk < Kd) {
            const int s = gk % d.S;
            const int r = (gk / d.S) % d.R;
            const int c = gk / (d.R * d.S);
            const int ih = ihb + r;
            const int iw = iwb + s;
            if (ih >= 0 && ih < d.H && iw >= 0 && iw < d.W) {
                v = x[idx4(n, c, ih, iw, d.C, d.H, d.W)];
            }
        }
        Bs[ty][tx] = v;
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < T; ++kk) {
            acc += As[ty][kk] * Bs[kk][tx];
        }
        __syncthreads();
    }

    if (row < M && colOk) {
        y[((size_t)n * d.K + row) * PQ + pq] = acc + (d.has_bias ? b[row] : 0.f);
    }
}

/**
 * @brief Implicit GEMM convolution with a TM x TN register tile per thread.
 *
 * Same GEMM view as conv2d_implicit_gemm, but each thread accumulates TM x TN outputs in
 * registers, and (c, r, s) plus each output column are decoded once per block into
 * shared-memory tables. Dynamic shared memory holds the reduction table (one int2 per
 * C*R*S index). Bias is added after summation, matching conv2d_implicit_gemm.
 *
 * @tparam BM Block tile height along K (output channels).
 * @tparam BN Block tile width along N*P*Q (output pixels).
 * @tparam BK Reduction chunk size.
 * @tparam TM Output channels per thread.
 * @tparam TN Output pixels per thread.
 */
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN)) conv2d_implicit_reg(
    ConvDims d,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y
) {
    constexpr int NT = (BM / TM) * (BN / TN);
    __shared__ float As[BK][BM + 4];  // +4 padding prevents bank conflicts
    __shared__ float Bs[BK][BN];
    __shared__ long long s_cbase[BN];  // input offset of each column's window origin
    __shared__ int s_ihb[BN];
    __shared__ int s_iwb[BN];
    extern __shared__ int2 s_ktab[];   // gk -> {c*H*W + r*W + s, (r << 16) | s}

    const int M = d.K;
    const int Kd = d.C * d.R * d.S;
    const int HW = d.H * d.W;
    const int PQ = d.P * d.Q;
    const long long Nc = (long long)d.N * PQ;
    const int tid = threadIdx.x;
    const int tRow = tid / (BN / TN);
    const int tCol = tid % (BN / TN);
    const int rowBase = blockIdx.y * BM;
    const long long colBase = (long long)blockIdx.x * BN;

    // Decode reduction indices once per block instead of on every tile load
    for (int gk = tid; gk < Kd; gk += NT) {
        const int s = gk % d.S;
        const int r = (gk / d.S) % d.R;
        const int c = gk / (d.R * d.S);
        s_ktab[gk] = make_int2(c * HW + r * d.W + s, (r << 16) | s);
    }

    // Decode this block's output columns once
    for (int nn = tid; nn < BN; nn += NT) {
        const long long col = colBase + nn;
        if (col < Nc) {
            const int n = (int)(col / PQ);
            const int pq = (int)(col % PQ);
            const int ihb = (pq / d.Q) * d.stride - d.pad;
            const int iwb = (pq % d.Q) * d.stride - d.pad;
            s_ihb[nn] = ihb;
            s_iwb[nn] = iwb;
            s_cbase[nn] = (long long)n * d.C * HW + (long long)ihb * d.W + iwb;
        } else {
            s_ihb[nn] = INT_MIN / 2;  // every bounds check fails, so the column loads zeros
            s_iwb[nn] = 0;
            s_cbase[nn] = 0;
        }
    }
    __syncthreads();

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.f;
        }
    }
    float ra[TM], rb[TN];

    for (int t = 0; t < Kd; t += BK) {
        for (int i = tid; i < BM * BK; i += NT) {
            const int m = i / BK;
            const int kk = i % BK;
            const int gr = rowBase + m;
            const int gk = t + kk;
            As[kk][m] = (gr < M && gk < Kd) ? w[(size_t)gr * Kd + gk] : 0.f;
        }
        // Gather input tile from table lookups; consecutive threads read consecutive pixels
        for (int i = tid; i < BK * BN; i += NT) {
            const int kk = i / BN;
            const int nn = i % BN;
            const int gk = t + kk;
            float v = 0.f;
            if (gk < Kd) {
                const int2 e = s_ktab[gk];
                const int ih = s_ihb[nn] + (e.y >> 16);
                const int iw = s_iwb[nn] + (e.y & 0xffff);
                if (ih >= 0 && ih < d.H && iw >= 0 && iw < d.W) {
                    v = x[s_cbase[nn] + e.x];
                }
            }
            Bs[kk][nn] = v;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                ra[i] = As[kk][tRow * TM + i];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                rb[j] = Bs[kk][tCol * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += ra[i] * rb[j];
                }
            }
        }
        __syncthreads();
    }

    // Bias after summation, matching conv2d_implicit_gemm
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const long long col = colBase + tCol * TN + j;
        if (col >= Nc) continue;
        const int n = (int)(col / PQ);
        const int pq = (int)(col % PQ);
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int r = rowBase + tRow * TM + i;
            if (r < M) {
                y[((size_t)n * d.K + r) * PQ + pq] = acc[i][j] + (d.has_bias ? b[r] : 0.f);
            }
        }
    }
}

/**
 * @brief Block tile height and width chosen from K so few output channels do not idle thread rows.
 */
inline int implicit_reg_bm(int K) { return K <= 4 ? 4 : K <= 16 ? 16 : K <= 32 ? 32 : 64; }
inline int implicit_reg_bn(int K) { return K <= 4 ? 256 : K <= 16 ? 128 : 64; }

/**
 * @brief Whether the reduction table plus static shared memory (at most ~13 KiB) fits in one block.
 */
inline bool implicit_reg_table_fits(const ConvDims& d, size_t smem_per_block) {
    return sizeof(int2) * (size_t)d.C * d.R * d.S + 16384 <= smem_per_block;
}

/**
 * @brief Auto selection for the implicit method: register tiles unless they would launch fewer
 * blocks than the GPU has SMs, the reduction table does not fit, or per-image input offsets
 * overflow the 32-bit table entries. Otherwise the original conv2d_implicit_gemm<16> runs.
 */
inline bool implicit_use_reg(const ConvDims& d, size_t smem_per_block, int sm_count) {
    if ((long long)d.C * d.H * d.W > INT_MAX || !implicit_reg_table_fits(d, smem_per_block)) {
        return false;
    }
    const long long cols = (long long)d.N * d.P * d.Q;
    const int bm = implicit_reg_bm(d.K);
    const int bn = implicit_reg_bn(d.K);
    return ((cols + bn - 1) / bn) * ((d.K + bm - 1) / bm) >= sm_count;
}

/**
 * @brief Launch conv2d_implicit_reg with one block per BM x BN output tile.
 */
template <int BM, int BN, int BK, int TM, int TN>
void launch_implicit_reg(const ConvDims& d, const float* xp, const float* wp, const float* bp, float* yp,
                         cudaStream_t stream) {
    const long long cols = (long long)d.N * d.P * d.Q;
    const size_t bytes = sizeof(int2) * d.C * d.R * d.S;
    dim3 grid((unsigned)((cols + BN - 1) / BN), (d.K + BM - 1) / BM);
    conv2d_implicit_reg<BM, BN, BK, TM, TN><<<grid, (BM / TM) * (BN / TN), bytes, stream>>>(d, xp, wp, bp, yp);
}

/**
 * @brief Launch the register-tiled configuration matching implicit_reg_bm / implicit_reg_bn for K.
 */
inline void launch_implicit_reg_for_k(const ConvDims& d, const float* xp, const float* wp, const float* bp,
                                      float* yp, cudaStream_t stream) {
    if (d.K <= 4) {
        launch_implicit_reg<4, 256, 8, 1, 4>(d, xp, wp, bp, yp, stream);
    } else if (d.K <= 16) {
        launch_implicit_reg<16, 128, 8, 4, 4>(d, xp, wp, bp, yp, stream);
    } else if (d.K <= 32) {
        launch_implicit_reg<32, 64, 8, 4, 4>(d, xp, wp, bp, yp, stream);
    } else {
        launch_implicit_reg<64, 64, 8, 4, 4>(d, xp, wp, bp, yp, stream);
    }
}

// ---------------------------------------------------------------------------
// 7. Grid-Stride Direct Convolution
// ---------------------------------------------------------------------------

/**
 * @brief Retrieve physical Streaming Multiprocessor (SM) ID via PTX assembly.
 */
__device__ __forceinline__ unsigned current_smid() {
#if defined(__CUDA_ARCH__)
    unsigned s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    return s;
#else
    return 0;
#endif
}

#define CONV_MAX_SMID 1024

/**
 * @brief Direct convolution with grid-stride loop for hardware occupancy scaling studies.
 *
 * Processes output elements via a fixed number of thread blocks executing grid-stride loops.
 * Optionally records active SM hardware IDs.
 *
 * @param d Convolution dimensional metadata.
 * @param x Input tensor [N, C, H, W] in device memory.
 * @param w Filter weights [K, C, R, S] in device memory.
 * @param b Bias vector [K] in device memory.
 * @param y Output tensor [N, K, P, Q] in device memory.
 * @param sm_flags Optional diagnostic array [CONV_MAX_SMID] recording touched SM indices.
 */
__global__ void conv2d_gridstride(
    ConvDims d,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int* __restrict__ sm_flags
) {
    if (sm_flags && threadIdx.x == 0) {
        const unsigned sid = current_smid();
        if (sid < CONV_MAX_SMID) {
            sm_flags[sid] = 1;
        }
    }

    const size_t total = (size_t)d.N * d.K * d.P * d.Q;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const int q = (int)(idx % d.Q);
        const int p = (int)((idx / d.Q) % d.P);
        const int k = (int)((idx / ((size_t)d.Q * d.P)) % d.K);
        const int n = (int)(idx / ((size_t)d.Q * d.P * d.K));
        float acc = d.has_bias ? b[k] : 0.f;

        for (int c = 0; c < d.C; ++c) {
            for (int r = 0; r < d.R; ++r) {
                const int ih = p * d.stride - d.pad + r;
                if (ih < 0 || ih >= d.H) continue;
                for (int s = 0; s < d.S; ++s) {
                    const int iw = q * d.stride - d.pad + s;
                    if (iw < 0 || iw >= d.W) continue;
                    acc += x[idx4(n, c, ih, iw, d.C, d.H, d.W)] * w[idx4(k, c, r, s, d.C, d.R, d.S)];
                }
            }
        }
        y[idx] = acc;
    }
}
