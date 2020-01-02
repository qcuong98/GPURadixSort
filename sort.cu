#include "main.h"

#define CTA_SIZE 4

__global__ void scanBlkKernel(uint32_t * in, int n, uint32_t * out, uint32_t * blkSums, int bit) {   
    extern __shared__ uint32_t s_in[];

    int id_in = CTA_SIZE * (blockDim.x * blockIdx.x + threadIdx.x);
    uint32_t val = 0;
    for (int i = 0; i < CTA_SIZE; ++i)
        val += (id_in + i < n ? (bit < 0 ? in[id_in + i] : ((in[id_in + i] >> bit) & 1)) : 0);
    s_in[threadIdx.x] = val;
    __syncthreads();

    int turn = 0;
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        turn ^= 1;
        int cur = s_in[threadIdx.x + (turn ^ 1) * blockDim.x];
        if (threadIdx.x >= stride)
            cur += s_in[threadIdx.x - stride + (turn ^ 1) * blockDim.x]; 
        s_in[threadIdx.x + turn * blockDim.x] = cur;
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1) { // last thread
        blkSums[blockIdx.x] = s_in[threadIdx.x + turn * blockDim.x];
    }

    val = s_in[threadIdx.x + turn * blockDim.x];
    for (int i = CTA_SIZE - 1; i >= 0; --i)
        if (id_in + i < n) {
            out[id_in + i] = val;
            val -= (id_in + i < n ? (bit < 0 ? in[id_in + i] : ((in[id_in + i] >> bit) & 1)) : 0);
        }
}

__global__ void sumPrefixBlkKernel(uint32_t * out, int n, uint32_t * blkSums) {
    int id_in = CTA_SIZE * (blockDim.x * blockIdx.x + threadIdx.x);
    for (int i = 0; i < CTA_SIZE; ++i)
        if (id_in + i < n && blockIdx.x > 0) {
            out[id_in + i] += blkSums[blockIdx.x - 1];
        }
}

void computeScanArray(uint32_t* d_in, uint32_t* d_out, int n, dim3 blkSize, int bit) {
    dim3 gridSize((n - 1) / (CTA_SIZE * blkSize.x) + 1);

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
            dst[n0 + histScan[i] - 1] = val;
        } else {
            dst[i - histScan[i]] = val;
        }
    }
}

void sort(const uint32_t * in, int n, uint32_t * out, int k, int * blockSizes) {
    uint32_t * d_src;
    uint32_t * d_dst;
    uint32_t * d_histScan;
    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_src, in, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_dst, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_histScan, n * sizeof(uint32_t)));

    // Compute block and grid size for scan and scatter phase
    dim3 blockSizeScan(blockSizes[1]);
    dim3 gridSizeScan((n - 1) / blockSizes[1] + 1);
    dim3 blockSizeScatter(blockSizes[2]);
    dim3 gridSizeScatter((n - 1) / blockSizes[2] + 1);

    for (int bit = 0; bit < sizeof(uint32_t) * 8; ++bit) {
        computeScanArray(d_src, d_histScan, n, blockSizeScan, bit);
        int n1;
        CHECK(cudaMemcpy(&n1, d_histScan + n - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        int n0 = n - n1;
        scatterKernel<<<gridSizeScatter, blockSizeScatter>>>(d_src, n, d_histScan, d_dst, bit, n0);
        CHECK(cudaDeviceSynchronize());
        
        uint32_t * tmp = d_src;
        d_src = d_dst;
        d_dst = tmp;
    }

    CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_histScan));
}
