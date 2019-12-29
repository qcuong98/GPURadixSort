#include "main.h"

__device__ uint32_t getBin(uint32_t val, uint32_t bit, uint32_t nBins) {
    return (val >> bit) & (nBins - 1);
}

__global__ void computeHistKernel(uint32_t * in, int n, uint32_t * hist, int nBins, int bit, int gridSize) {
    extern __shared__ int s_hist[];
    for (int idx = threadIdx.x; idx < nBins; idx += blockDim.x)
        s_hist[idx] = 0;
    __syncthreads();

    // Each block computes its local hist using atomic on SMEM
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        atomicAdd(&s_hist[getBin(in[i], bit, nBins)], 1);
    __syncthreads();

    // Each block adds its local hist to global hist using atomic on GMEM
    for (int digit = threadIdx.x; digit < nBins; digit += blockDim.x)
        hist[blockIdx.x + digit * gridSize] = s_hist[digit];
}

__global__ void scanBlkKernel(uint32_t * in, int n, uint32_t * out, uint32_t * blkSums) {
    extern __shared__ uint32_t s_in[];
    int id_in = blockDim.x * blockIdx.x + threadIdx.x;
    s_in[threadIdx.x] = id_in < n ? in[id_in] : 0;
    __syncthreads();

    int turn = 0;
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        turn ^= 1;
        uint32_t cur = s_in[threadIdx.x + (turn ^ 1) * blockDim.x];
        if (threadIdx.x >= stride)
            cur += s_in[threadIdx.x - stride + (turn ^ 1) * blockDim.x]; 
        s_in[threadIdx.x + turn * blockDim.x] = cur;
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1) { // last thread
        blkSums[blockIdx.x] = s_in[threadIdx.x + turn * blockDim.x];
    }

    if (id_in < n) {
        out[id_in] = s_in[threadIdx.x + turn * blockDim.x];
    }
}

__global__ void sumPrefixBlkKernel(uint32_t * out, int n, uint32_t * blkSums) {
    int id_in = blockDim.x * blockIdx.x + threadIdx.x;
    if (id_in < n && blockIdx.x > 0) {
        out[id_in] += blkSums[blockIdx.x - 1];
    }
}

__global__ void reduceKernel(uint32_t * in, int n, uint32_t * out) {
    int id_in = blockDim.x * blockIdx.x + threadIdx.x;
    if (id_in < n)
        out[id_in] -= in[id_in];
}

void computeScanArray(uint32_t* d_in, uint32_t* d_out, int n, dim3 blkSize) {
    dim3 gridSize((n - 1) / blkSize.x + 1);

    uint32_t * d_blkSums;
    CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(uint32_t)));
    uint32_t * d_sum_blkSums;
    CHECK(cudaMalloc(&d_sum_blkSums, gridSize.x * sizeof(uint32_t)));

    scanBlkKernel<<<gridSize, blkSize, 2 * blkSize.x * sizeof(uint32_t)>>>
        (d_in, n, d_out, d_blkSums);
    if (gridSize.x != 1) {
        computeScanArray(d_blkSums, d_sum_blkSums, gridSize.x, blkSize);
    }
    sumPrefixBlkKernel<<<gridSize, blkSize>>>(d_out, n, d_sum_blkSums);

    CHECK(cudaFree(d_sum_blkSums));
    CHECK(cudaFree(d_blkSums));
}

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) + ((n) >> LOG_NUM_BANKS))

__device__ void sortLocal(uint32_t* src, uint32_t* scan, int bit, int k, int ai, int bi) {
    for (int blockBit = bit; blockBit < bit + k; ++blockBit) {
        uint32_t val_a = src[CONFLICT_FREE_OFFSET(ai)];
        uint32_t val_b = src[CONFLICT_FREE_OFFSET(bi)];

        // compute scan
        scan[CONFLICT_FREE_OFFSET(ai)] = (src[CONFLICT_FREE_OFFSET(ai)] >> blockBit) & 1;
        scan[CONFLICT_FREE_OFFSET(bi)] = (src[CONFLICT_FREE_OFFSET(bi)] >> blockBit) & 1;
        __syncthreads();

        // reduction phase
        for (int stride = 1, d = blockDim.x; stride <= blockDim.x; stride <<= 1, d >>= 1) {
            if (threadIdx.x < d) {
                int cur = 2 * stride * (threadIdx.x + 1) - 1;
                int prev = cur - stride;
                scan[CONFLICT_FREE_OFFSET(cur)] += scan[CONFLICT_FREE_OFFSET(prev)];
            }
            __syncthreads();
        }
        // post-reduction phase
        for (int stride = blockDim.x >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1) {
            if (threadIdx.x < d - 1) {
                int prev = 2 * stride * (threadIdx.x + 1) - 1;
                int cur = prev + stride;
                scan[CONFLICT_FREE_OFFSET(cur)] += scan[CONFLICT_FREE_OFFSET(prev)];
            }
            __syncthreads();
        }
        
        // scatter
        int n0 = 2 * blockDim.x - scan[CONFLICT_FREE_OFFSET(2 * blockDim.x - 1)];
        
        if ((val_a >> blockBit) & 1)
            src[CONFLICT_FREE_OFFSET(n0 + scan[CONFLICT_FREE_OFFSET(ai)] - 1)] = val_a;
        else
            src[CONFLICT_FREE_OFFSET(ai - scan[CONFLICT_FREE_OFFSET(ai)])] = val_a;

        if ((val_b >> blockBit) & 1)
            src[CONFLICT_FREE_OFFSET(n0 + scan[CONFLICT_FREE_OFFSET(bi)] - 1)] = val_b;
        else
            src[CONFLICT_FREE_OFFSET(bi - scan[CONFLICT_FREE_OFFSET(bi)])] = val_b;
        
        __syncthreads();
    }
}

__device__ void countEqualBefore(uint32_t* src, uint32_t* buffer, int bit, int nBins, int ai, int bi) {
    buffer[CONFLICT_FREE_OFFSET(ai)] = 1;
    buffer[CONFLICT_FREE_OFFSET(bi)] = 1;
    __syncthreads();
    // reduction phase
    for (int stride = 1, d = blockDim.x; stride <= blockDim.x; stride <<= 1, d >>= 1) {
        int cur = 2 * stride * (threadIdx.x + 1) - 1;
        int prev = cur - stride;
        if (threadIdx.x < d && getBin(src[CONFLICT_FREE_OFFSET(cur)], bit, nBins) == getBin(src[CONFLICT_FREE_OFFSET(prev)], bit, nBins)) {
            buffer[CONFLICT_FREE_OFFSET(cur)] += buffer[CONFLICT_FREE_OFFSET(prev)];
        }
        __syncthreads();
    }
    // post-reduction phase
    for (int stride = blockDim.x >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1) {
        int cur = 2 * stride * (threadIdx.x + 1) + stride - 1;
        int prev = cur - stride;
        if (threadIdx.x < d - 1 && getBin(src[CONFLICT_FREE_OFFSET(cur)], bit, nBins) == getBin(src[CONFLICT_FREE_OFFSET(prev)], bit, nBins)) {
            buffer[CONFLICT_FREE_OFFSET(cur)] += buffer[CONFLICT_FREE_OFFSET(prev)];
        }
        __syncthreads();
    }
}

__global__ void scatterKernel(uint32_t* src, int n, uint32_t* histScan, int k, int nBins, uint32_t* dst, int bit, int gridSize) {
    extern __shared__ uint32_t s[];
    uint32_t* localSrc = s;
    uint32_t* localScan = localSrc + 3 * blockDim.x;

    int id_ai = 2 * blockDim.x * blockIdx.x + 2 * threadIdx.x;
    int id_bi = 2 * blockDim.x * blockIdx.x + 2 * threadIdx.x + 1;
    int ai = 2 * threadIdx.x;
    int bi = 2 * threadIdx.x + 1;
    localSrc[CONFLICT_FREE_OFFSET(ai)] = id_ai < n ? src[id_ai] : UINT_MAX;
    localSrc[CONFLICT_FREE_OFFSET(bi)] = id_bi < n ? src[id_bi] : UINT_MAX;
    __syncthreads();

    // sort locally using radix sort with k = 1
    sortLocal(localSrc, localScan, bit, k, ai, bi);

    // count equals before
    countEqualBefore(localSrc, localScan, bit, nBins, ai, bi); 
    
    // scatter
    uint32_t pos;
    pos = histScan[blockIdx.x + getBin(localSrc[CONFLICT_FREE_OFFSET(ai)], bit, nBins) * gridSize] + localScan[CONFLICT_FREE_OFFSET(ai)] - 1;
    if (pos < n) {
        dst[pos] = localSrc[CONFLICT_FREE_OFFSET(ai)];
    }
    pos = histScan[blockIdx.x + getBin(localSrc[CONFLICT_FREE_OFFSET(bi)], bit, nBins) * gridSize] + localScan[CONFLICT_FREE_OFFSET(bi)] - 1;
    if (pos < n) {
        dst[pos] = localSrc[CONFLICT_FREE_OFFSET(bi)];
    }
    // pos = histScan[blockIdx.x + getBin(localSrc[bi], bit, nBins) * gridSize] + localScan[bi] - 1;
    // if (pos < n) {
    //     dst[pos] = localSrc[bi];
    // }
}

void sort(const uint32_t * in, int n, uint32_t * out, int k, int * blockSizes) {
    int nBins = 1 << k;
    uint32_t * d_src;
    uint32_t * d_dst;
    uint32_t * d_hist;
    uint32_t * d_histScan;
    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_src, in, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_dst, n * sizeof(uint32_t)));

    // Compute block and grid size for scan and scatter phase
    dim3 blockSizeHist(blockSizes[0]);
    dim3 gridSizeHist((n - 1) / blockSizeHist.x + 1);
    dim3 blockSizeScan(blockSizes[1]);
    dim3 gridSizeScan((n - 1) / blockSizeScan.x + 1);
    dim3 blockSizeScatter(blockSizes[2]);
    dim3 gridSizeScatter((n - 1) / (2 * blockSizeScatter.x) + 1);

    int histSize = nBins * gridSizeHist.x;
    CHECK(cudaMalloc(&d_hist, histSize * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_histScan, histSize * sizeof(uint32_t)));

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += k) {
        // compute hist
        computeHistKernel<<<gridSizeHist, blockSizeHist, nBins * sizeof(uint32_t)>>>
            (d_src, n, d_hist, nBins, bit, gridSizeHist.x);
        
        // compute hist scan
        computeScanArray(d_hist, d_histScan, histSize, blockSizeScan);
        reduceKernel<<<gridSizeScan, blockSizeScan>>>
            (d_hist, histSize, d_histScan);
        
        // scatter
        scatterKernel<<<gridSizeScatter, blockSizeScatter, (6 * blockSizeScatter.x) * sizeof(uint32_t)>>>
            (d_src, n, d_histScan, k, nBins, d_dst, bit, gridSizeHist.x);
        
        uint32_t * tmp = d_src; d_src = d_dst; d_dst = tmp;
    }

    CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_histScan));
}
