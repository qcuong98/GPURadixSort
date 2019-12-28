#include "main.h"

__device__ uint32_t getBin(uint32_t val, uint32_t bit, uint32_t nBins) {
    return (val >> bit) & (nBins - 1);
}

__global__ void computeHistKernel(uint32_t * in, int n, uint32_t * hist, int nBins, int bit, int gridSize) {
    extern __shared__ uint32_t s_hist[];
    for (int idx = threadIdx.x; idx < nBins; idx += blockDim.x)
        s_hist[idx] = 0;
    __syncthreads();

    // Each block computes its local hist using atomic on SMEM
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t val = i < n ? in[i] : UINT_MAX;
    atomicAdd(&s_hist[getBin(val, bit, nBins)], 1);
    __syncthreads();

    // Each block adds its local hist to global hist using atomic on GMEM
    for (uint32_t digit = threadIdx.x; digit < nBins; digit += blockDim.x) {
        hist[blockIdx.x + digit * gridSize] = s_hist[digit];
    }
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

__device__ void sortLocal(uint32_t* src, uint32_t* scan, int bit, int k) {
    for (int blockBit = bit; blockBit < bit + k; ++blockBit) {
        uint32_t val = src[threadIdx.x];
        // compute scan
        scan[threadIdx.x] = (src[threadIdx.x] >> blockBit) & 1;
        __syncthreads();
        int turn = 0;
        for (int stride = 1; stride < blockDim.x; stride <<= 1) {
            turn ^= 1;
            uint32_t cur = scan[threadIdx.x + (turn ^ 1) * blockDim.x];
            if (threadIdx.x >= stride)
                cur += scan[threadIdx.x - stride + (turn ^ 1) * blockDim.x]; 
            scan[threadIdx.x + turn * blockDim.x] = cur;
            __syncthreads();
        }
        
        // scatter
        int n0 = blockDim.x - scan[blockDim.x - 1 + turn * blockDim.x];
        
        if ((val >> blockBit) & 1)
            src[n0 + scan[threadIdx.x + turn * blockDim.x] - 1] = val;
        else
            src[threadIdx.x - scan[threadIdx.x + turn * blockDim.x]] = val;
        
        __syncthreads();
    }
}

__device__ uint32_t* countEqualBefore(uint32_t* src, uint32_t* buffer, int bit, int nBins) {
    uint32_t thisSrc = src[threadIdx.x], thisBin = getBin(thisSrc, bit, nBins);
    buffer[threadIdx.x] = 1;
    __syncthreads();
    int turn = 0;
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        turn ^= 1;
        uint32_t cur = buffer[threadIdx.x + (turn ^ 1) * blockDim.x];
        if (threadIdx.x >= stride &&  thisBin == getBin(src[threadIdx.x - stride], bit, nBins))
            cur += buffer[threadIdx.x - stride + (turn ^ 1) * blockDim.x]; 
        buffer[threadIdx.x + turn * blockDim.x] = cur;
        __syncthreads();
    }
    return buffer + turn * blockDim.x;
}

__global__ void scatterKernel(uint32_t* src, int n, uint32_t* histScan, int k, int nBins, uint32_t* dst, int bit, int gridSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ uint32_t s[];
    uint32_t* localSrc = s;
    uint32_t* localScan = localSrc + blockDim.x;
    
    localSrc[threadIdx.x] = i < n ? src[i] : UINT_MAX;

    // sort locally using radix sort with k = 1
    sortLocal(localSrc, localScan, bit, k);

    // count equals before
    uint32_t* count = countEqualBefore(localSrc, localScan, bit, nBins); 
    
    // scatter
    uint32_t pos =
        histScan[blockIdx.x * nBins + getBin(localSrc[threadIdx.x], bit, nBins)]
        + count[threadIdx.x]
        - 1;
    
    if (pos < n) {
        dst[pos] = localSrc[threadIdx.x];
    }
}

__global__ void transpose(uint32_t *iMatrix, uint32_t *oMatrix, int rows, int cols)
{
    __shared__ int s_blkData[32][33];
    // Each block load data efficiently from GMEM to SMEM
    int iR = blockIdx.x * blockDim.x + threadIdx.y;
    int iC = blockIdx.y * blockDim.y + threadIdx.x;
    s_blkData[threadIdx.y][threadIdx.x] = (iR < rows && iC < cols) ? iMatrix[iR * cols + iC] : 0;
    __syncthreads();
    // Each block write data efficiently from SMEM to GMEM
    int oR = blockIdx.y * blockDim.y + threadIdx.y;
    int oC = blockIdx.x * blockDim.x + threadIdx.x;
    if (oR < cols && oC < rows)
        oMatrix[oR * rows + oC] = s_blkData[threadIdx.x][threadIdx.y];
}

void printArr(const char* tag, uint32_t* d_arr, int size) {
  uint32_t* arr = (uint32_t*)malloc(size * sizeof(uint32_t));
  cudaMemcpy(arr, d_arr, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  printf("%s, size = %d\n", tag, size);
  for (int i = 0; i < size; ++i) {
    printf("%u ", (unsigned)arr[i]);
  }
  printf("\n");
  free(arr);
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
    dim3 gridSizeScatter((n - 1) / blockSizeScatter.x + 1);
    dim3 blockSizeTranspose(32, 32);
    dim3 gridSizeTransposeHist((gridSizeHist.x - 1) / blockSizeTranspose.x + 1, (gridSizeHist.x - 1) / blockSizeTranspose.x + 1);
    dim3 gridSizeTransposeHistScan((nBins - 1) / blockSizeTranspose.x + 1, (gridSizeHist.x - 1) / blockSizeTranspose.x + 1);
    
    int histSize = nBins * gridSizeHist.x;
    CHECK(cudaMalloc(&d_hist, 1 * histSize * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_histScan, 2 * histSize * sizeof(uint32_t)));
    
    dim3 blockSizeReduce = blockSizeScan;
    dim3 gridSizeReduce((histSize - 1) / blockSizeReduce.x + 1);
    
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += k) {
//       printf("bit = %d\n", bit);
        // compute hist
        computeHistKernel<<<gridSizeHist, blockSizeHist, nBins * sizeof(uint32_t)>>>
            (d_src, n, d_hist, nBins, bit, gridSizeHist.x);
//         transpose<<<gridSizeTransposeHist, blockSizeTranspose>>>
//           (d_hist + histSize, d_hist, gridSizeHist.x, nBins);
//         printArr("src", d_src, n);
//         printArr("hist", d_hist, histSize);
        
        // compute hist scan
        computeScanArray(d_hist, d_histScan + histSize, histSize, blockSizeScan);
        reduceKernel<<<gridSizeReduce, blockSizeReduce>>>
            (d_hist, histSize, d_histScan + histSize);
//         printArr("histScan after reduce", d_histScan, histSize);
        
        transpose<<<gridSizeTransposeHistScan, blockSizeTranspose>>>
          (d_histScan + histSize, d_histScan, nBins, gridSizeHist.x);
        // scatter
        scatterKernel<<<gridSizeScatter, blockSizeScatter, (3 * blockSizeScatter.x) * sizeof(uint32_t)>>>
            (d_src, n, d_histScan, k, nBins, d_dst, bit, gridSizeHist.x);
//         printArr("dst", d_dst, n);
        
        uint32_t * tmp = d_src; d_src = d_dst; d_dst = tmp;
    }

    CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_histScan));
}
