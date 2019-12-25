#include "main.h"

__global__ void computeHistKernel2(int * in, int n, int * hist, int nBins, int bit) {
    extern __shared__ int s_hist[];
    for (int idx = threadIdx.x; idx < nBins; idx += blockDim.x)
        s_hist[idx] = 0;
    __syncthreads();

    // Each block computes its local hist using atomic on SMEM
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        atomicAdd(&s_hist[(in[i] >> bit) & (nBins - 1)], 1);
    __syncthreads();

    // Each block adds its local hist to global hist using atomic on GMEM
    for (int idx = threadIdx.x; idx < nBins; idx += blockDim.x)
        atomicAdd(&hist[idx], s_hist[idx]);
}

__global__ void scanBlkKernel(uint32_t * in, int n, uint32_t * out, uint32_t * blkSums, int bit) {   
    extern __shared__ uint32_t s_in[];
    int id_in = blockDim.x * blockIdx.x + threadIdx.x;
    s_in[threadIdx.x] = id_in < n ? (bit < 0 ? in[id_in] : ((in[id_in] >> bit) & 1)) : 0;
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

    if (id_in < n)
        out[id_in] = s_in[threadIdx.x + turn * blockDim.x];
}

__global__ void sumPrefixBlkKernel(uint32_t * out, int n, uint32_t * blkSums) {
    int id_in = blockDim.x * blockIdx.x + threadIdx.x;
    if (id_in < n && blockIdx.x > 0) {
        out[id_in] += blkSums[blockIdx.x - 1];
    }
}

__global__ void reduceKernel(uint32_t * in, int n, uint32_t * out, int bit) {
    int id_in = blockDim.x * blockIdx.x + threadIdx.x;
    if (id_in < n)
        out[id_in] -= ((in[id_in] >> bit) & 1);
}

void computeScanArray(uint32_t* d_in, uint32_t* d_out, int n, dim3 blkSize, int bit) {
    dim3 gridSize((n - 1) / blkSize.x + 1);

    uint32_t * d_blkSums;
    CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(int)));
    uint32_t * d_sum_blkSums;
    CHECK(cudaMalloc(&d_sum_blkSums, gridSize.x * sizeof(int)));

    scanBlkKernel<<<gridSize, blkSize, 2 * blkSize.x * sizeof(int)>>>
        (d_in, n, d_out, d_blkSums, bit);
    if (gridSize.x != 1) {
        computeScanArray(d_blkSums, d_sum_blkSums, gridSize.x, blkSize, -1);
    }
    sumPrefixBlkKernel<<<gridSize, blkSize>>>(d_out, n, d_sum_blkSums);

    CHECK(cudaFree(d_sum_blkSums));
    CHECK(cudaFree(d_blkSums));
}

__global__ void scatterKernel(uint32_t* src, int n, uint32_t* histScan, uint32_t* dst, int bit) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        if ((src[i] >> bit) & 1) {
            int n0 = n - histScan[n - 1] - ((src[n - 1] >> bit) & 1);
            dst[n0 + histScan[i]] = src[i];
        }
        else
            dst[i - histScan[i]] = src[i];
    }
}
void sort(const uint32_t * in, int n, uint32_t * out, int k, int * blockSizes) {
    uint32_t * histScan = (uint32_t *) malloc(n * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later

    uint32_t * d_src;
    uint32_t * d_dst;
    uint32_t * d_histScan;
    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_src, src, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_dst, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_histScan, n * sizeof(int)));

    // Compute block and grid size for histogram phase
    dim3 blockSizeHist(blockSizes[0]);
    dim3 gridSizeHist((n - 1) / blockSizes[0] + 1);
    dim3 blockSizeScan(blockSizes[1]);
    dim3 gridSizeScan((n - 1) / blockSizes[1] + 1);
    dim3 blockSizeScatter = blockSizeScan;
    dim3 gridSizeScatter((n - 1) / blockSizeScatter.x + 1);

    for (int bit = 0; bit < sizeof(uint32_t) * 8; ++bit) {
        computeScanArray(d_src, d_histScan, n, blockSizeScan, bit);
        reduceKernel<<<gridSizeScan, blockSizeScan>>>(d_src, n, d_histScan, bit);
        scatterKernel<<<gridSizeScatter, blockSizeScatter>>>(d_src, n, d_histScan, d_dst, bit);
        CHECK(cudaDeviceSynchronize());
        
        uint32_t * tmp = d_src;
        d_src = d_dst;
        d_dst = tmp;
    }

    CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    free(histScan);
    free(originalSrc);
    
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_histScan));
}
