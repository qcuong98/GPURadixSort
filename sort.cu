#include "main.h"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
        ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 
        
__global__ void scanBlkKernel(uint32_t * in, int n, uint32_t * out, uint32_t * blkSums, int bit) {   
    extern __shared__ uint32_t s_in[];
    int id_ai = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    int id_bi = 2 * blockDim.x * blockIdx.x + threadIdx.x + blockDim.x;
    int ai = threadIdx.x;
    int bi = threadIdx.x + blockDim.x;
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    s_in[ai + bankOffsetA] = id_ai < n ? (bit < 0 ? in[id_ai] : ((in[id_ai] >> bit) & 1)) : 0;
    s_in[bi + bankOffsetB] = id_bi < n ? (bit < 0 ? in[id_bi] : ((in[id_bi] >> bit) & 1)) : 0;
    __syncthreads();

    // reduction phase
    for (int stride = 1; stride <= blockDim.x; stride <<= 1) {
        if (threadIdx.x < blockDim.x / stride) {
            int pos = 2 * stride * (threadIdx.x + 1) - 1;
            s_in[pos + CONFLICT_FREE_OFFSET(pos)] += s_in[pos - stride + CONFLICT_FREE_OFFSET(pos - stride)];
        }
        __syncthreads();
    }
    // post-reduction phase
    for (int stride = blockDim.x >> 1; stride >= 1; stride >>= 1) {
        if (threadIdx.x < blockDim.x / stride - 1) {
            int pos = 2 * stride * (threadIdx.x + 1) + stride - 1;
            s_in[pos + CONFLICT_FREE_OFFSET(pos)] += s_in[pos - stride + CONFLICT_FREE_OFFSET(pos - stride)];
        }
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1) { // last thread
        blkSums[blockIdx.x] = s_in[bi + bankOffsetB];
    }

    if (id_ai < n) {
        out[id_ai] = s_in[ai + bankOffsetA];
    }
    if (id_bi < n) {
        out[id_bi] = s_in[bi + bankOffsetB];
    }
}

__global__ void sumPrefixBlkKernel(uint32_t * out, int n, uint32_t * blkSums) {
    int id_in = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
    if (blockIdx.x > 0) {
        if (id_in < n) {
            out[id_in] += blkSums[blockIdx.x - 1];
        }
        if (id_in + 1 < n) {
            out[id_in + 1] += blkSums[blockIdx.x - 1];
        }
    }
}

void computeScanArray(uint32_t* d_in, uint32_t* d_out, int n, dim3 blkSize, int bit) {
    dim3 gridSize((n - 1) / (2 * blkSize.x) + 1);

    uint32_t * d_blkSums;
    CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(uint32_t)));
    uint32_t * d_sum_blkSums;
    CHECK(cudaMalloc(&d_sum_blkSums, gridSize.x * sizeof(uint32_t)));

    scanBlkKernel<<<gridSize, blkSize, 2 * blkSize.x * sizeof(uint32_t)>>>
        (d_in, n, d_out, d_blkSums, bit);
    if (gridSize.x != 1) {
        computeScanArray(d_blkSums, d_sum_blkSums, gridSize.x, blkSize, -1);
    }
    sumPrefixBlkKernel<<<gridSize, blkSize>>>(d_out, n, d_sum_blkSums);

    CHECK(cudaFree(d_sum_blkSums));
    CHECK(cudaFree(d_blkSums));
}

__global__ void scatterKernel(uint32_t* src, int n, uint32_t* histScan, uint32_t* dst, int bit, int n0) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        int val = src[i];
        if (val >> bit & 1) {
            dst[n0 + histScan[i]] = val;
        }
        else {
            dst[i - histScan[i]] = val;
        }
    }
}

void sort(const uint32_t * in, int n, uint32_t * out, int k, int * blockSizes) {
    uint32_t * histScan = (uint32_t *) malloc(n * sizeof(uint32_t));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later

    uint32_t * d_src;
    uint32_t * d_dst;
    uint32_t * d_histScan;
    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_src, src, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_dst, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_histScan, n * sizeof(uint32_t)));

    // Compute block and grid size for scan and scatter phase
    dim3 blockSizeScan(blockSizes[1]);
    dim3 blockSizeScatter(blockSizes[2]);
    dim3 gridSizeScatter((n - 1) / blockSizes[2] + 1);

    for (int bit = 0; bit < sizeof(uint32_t) * 8; ++bit) {
        computeScanArray(d_src, d_histScan, n, blockSizeScan, bit);
        int n1;
        CHECK(cudaMemcpy(&n1, d_histScan + n - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        int n0 = n - n1 - 1;
        scatterKernel<<<gridSizeScatter, blockSizeScatter>>>(d_src, n, d_histScan, d_dst, bit, n0);
        CHECK(cudaDeviceSynchronize());
        
        uint32_t * tmp = d_src;
        d_src = d_dst;
        d_dst = tmp;
    }

    CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    free(histScan);
    free(originalSrc);
    
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_histScan));
}
