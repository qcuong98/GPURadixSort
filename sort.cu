#include "main.h"

__global__ void computeHistKernel2(uint32_t * in, int n, int * hist, int nBins, int bit) {
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

__global__ void scanBlkKernel(int * in, int n, int * out, int * blkSums) {   
    extern __shared__ int s_in[];
    int id_in = blockDim.x * blockIdx.x + threadIdx.x;
    s_in[threadIdx.x] = id_in < n ? in[id_in] : 0;
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

__global__ void sumPrefixBlkKernel(int * out, int n, int * blkSums) {
    int id_in = blockDim.x * blockIdx.x + threadIdx.x;
    if (id_in < n && blockIdx.x > 0) {
        out[id_in] += blkSums[blockIdx.x - 1];
    }
}

__global__ void reduceKernel(int * in, int n, int * out) {
    int id_in = blockDim.x * blockIdx.x + threadIdx.x;
    if (id_in < n)
        out[id_in] -= in[id_in];
}

void computeScanArray(int* d_in, int* d_out, int n, dim3 blkSize) {
    dim3 gridSize((n - 1) / blkSize.x + 1);

    int * d_blkSums;
    CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(int)));
    int * d_sum_blkSums;
    CHECK(cudaMalloc(&d_sum_blkSums, gridSize.x * sizeof(int)));

    scanBlkKernel<<<gridSize, blkSize, 2 * blkSize.x * sizeof(int)>>>(d_in, n, d_out, d_blkSums);
    if (gridSize.x != 1) {
        computeScanArray(d_blkSums, d_sum_blkSums, gridSize.x, blkSize);
    }
    sumPrefixBlkKernel<<<gridSize, blkSize>>>(d_out, n, d_sum_blkSums);

    CHECK(cudaFree(d_sum_blkSums));
    CHECK(cudaFree(d_blkSums));
}

void sort(const uint32_t * in, int n, uint32_t * out, int k, int * blockSizes) {
    int nBins = 1 << k; // 2^k
    int * hist = (int *) malloc(nBins * sizeof(int));
    int * histScan = (int *) malloc(nBins * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;

    uint32_t * d_src;
    int * d_hist, * d_histScan;
    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_hist, nBins * sizeof(int)));
    CHECK(cudaMalloc(&d_histScan, nBins * sizeof(int)));

    // Compute block and grid size for histogram phase
    dim3 blockSizeHist(blockSizes[0]);
    dim3 gridSizeHist((n - 1) / blockSizes[0] + 1);
    dim3 blockSizeScan(blockSizes[1]);
    dim3 gridSizeScan((n - 1) / blockSizes[1] + 1);
    size_t smemSizeHist = nBins * sizeof(int);

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += k) {
        CHECK(cudaMemcpy(d_src, src, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_hist, 0, nBins * sizeof(int)));

        computeHistKernel2<<<gridSizeHist, blockSizeHist, smemSizeHist>>>
            (d_src, n, d_hist, nBins, bit);
        computeScanArray(d_hist, d_histScan, nBins, blockSizeScan);
        reduceKernel<<<gridSizeScan, blockSizeScan>>>(d_hist, nBins, d_histScan);
        CHECK(cudaMemcpy(histScan, d_histScan, nBins * sizeof(int), cudaMemcpyDeviceToHost));

        for (int i = 0; i < n; ++i) {
            int bin = (src[i] >> bit) & (nBins - 1);
            dst[histScan[bin]] = src[i];
            ++histScan[bin];
        }

        uint32_t * tmp = src;
        src = dst;
        dst = tmp;
    }

    memcpy(out, src, n * sizeof(uint32_t));

    free(histScan);
    free(originalSrc);

    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_histScan));
}