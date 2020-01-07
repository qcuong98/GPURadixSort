#include "main.h"

#define CTA_SIZE 4

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) + ((n) >> LOG_NUM_BANKS))

#define BLOCKDIM_SCAN 128
#define BLOCKDIM_SORTLOCAL 128

__device__ uint32_t getBin(uint32_t val, uint32_t bit, uint32_t nBins) {
    return (val >> bit) & (nBins - 1);
}

__global__ void computeHistKernel(uint32_t * in, int n, uint32_t * hist, int nBins, int bit, int gridSize, uint32_t* count) {
    extern __shared__ uint32_t s[];
    uint32_t* s_hist = s;
    uint32_t* s_in = s_hist + CONFLICT_FREE_OFFSET(nBins);
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = threadIdx.x; idx < nBins; idx += blockDim.x)
        s_hist[CONFLICT_FREE_OFFSET(idx)] = 0;
    s_in[CONFLICT_FREE_OFFSET(threadIdx.x)] = i < n ? in[i] : UINT_MAX;
    __syncthreads();

    // Each block computes its local hist using atomic on SMEM
    if (i < n) {
        uint32_t val = s_in[CONFLICT_FREE_OFFSET(threadIdx.x)], thisBin = getBin(val, bit, nBins);
        if (i == n - 1 || threadIdx.x == blockDim.x - 1 || thisBin != getBin(s_in[CONFLICT_FREE_OFFSET(threadIdx.x + 1)], bit, nBins))
            s_hist[CONFLICT_FREE_OFFSET(thisBin)] = count[i];
    }
    __syncthreads();

    // Each block adds its local hist to global hist using atomic on GMEM
    for (int digit = threadIdx.x; digit < nBins; digit += blockDim.x)
        hist[blockIdx.x * nBins + digit] = s_hist[CONFLICT_FREE_OFFSET(digit)];
}

__global__ void scanBlkKernel(uint32_t * src, int n, uint32_t * out, uint32_t * blkSums) {
    extern __shared__ uint32_t s[];
    uint32_t* localScan = s;
    uint32_t* localScanCTA = localScan + CONFLICT_FREE_OFFSET(2 * CTA_SIZE * blockDim.x);

    int ai = threadIdx.x;
    int bi = threadIdx.x + blockDim.x;
    
    for (int i = threadIdx.x; i < 2 * CTA_SIZE * blockDim.x; i += blockDim.x) {
        int pos = (2 * CTA_SIZE * blockDim.x) * blockIdx.x + i;
        localScan[CONFLICT_FREE_OFFSET(i)] = pos < n ? src[pos] : 0;
    }
    __syncthreads();

    uint32_t tempA[CTA_SIZE], tempB[CTA_SIZE];
    # pragma unroll
    for (int i = 0; i < CTA_SIZE; ++i) {
        tempA[i] = localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)]; 
        tempB[i] = localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)]; 
        if (i) {
            tempA[i] += tempA[i - 1];
            tempB[i] += tempB[i - 1];
        }
    }

    // compute scan
    localScanCTA[CONFLICT_FREE_OFFSET(ai)] = tempA[CTA_SIZE - 1];
    localScanCTA[CONFLICT_FREE_OFFSET(bi)] = tempB[CTA_SIZE - 1];
    __syncthreads();

    // reduction phase
    # pragma unroll
    for (int stride = 1, d = BLOCKDIM_SCAN; stride <= BLOCKDIM_SCAN; stride <<= 1, d >>= 1) {
        if (threadIdx.x < d) {
            int cur = 2 * stride * (threadIdx.x + 1) - 1;
            int prev = cur - stride;
            localScanCTA[CONFLICT_FREE_OFFSET(cur)] += localScanCTA[CONFLICT_FREE_OFFSET(prev)];
        }
        __syncthreads();
    }
    // post-reduction phase
    # pragma unroll
    for (int stride = BLOCKDIM_SCAN >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1) {
        if (threadIdx.x < d - 1) {
            int prev = 2 * stride * (threadIdx.x + 1) - 1;
            int cur = prev + stride;
            localScanCTA[CONFLICT_FREE_OFFSET(cur)] += localScanCTA[CONFLICT_FREE_OFFSET(prev)];
        }
        __syncthreads();
    }
    
    uint32_t lastScanA = ai ? localScanCTA[CONFLICT_FREE_OFFSET(ai - 1)] : 0;
    uint32_t lastScanB = localScanCTA[CONFLICT_FREE_OFFSET(bi - 1)];
    __syncthreads();
    
    #pragma unroll
    for (int i = 0; i < CTA_SIZE; ++i) {
        tempA[i] += lastScanA;
        tempB[i] += lastScanB;
        
        localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)] = tempA[i];
        localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)] = tempB[i];
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < 2 * CTA_SIZE * blockDim.x; i += blockDim.x) {
        int pos = 2 * CTA_SIZE * blockDim.x * blockIdx.x + i;
        if (pos < n)
            out[pos] = localScan[CONFLICT_FREE_OFFSET(i)];
    }
    
    if (threadIdx.x == blockDim.x - 1) {
        blkSums[blockIdx.x] = tempB[CTA_SIZE - 1];
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

void computeScanArray(uint32_t* d_in, uint32_t* d_out, int n, dim3 blkSize, dim3 blockSizeCTA2) {
    dim3 gridSize((n - 1) / blkSize.x + 1);

    uint32_t * d_blkSums;
    CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(uint32_t)));
    uint32_t * d_sum_blkSums;
    CHECK(cudaMalloc(&d_sum_blkSums, gridSize.x * sizeof(uint32_t)));

    scanBlkKernel<<<gridSize, blockSizeCTA2, CONFLICT_FREE_OFFSET((2 * CTA_SIZE + 2) * blockSizeCTA2.x) * sizeof(uint32_t)>>>
        (d_in, n, d_out, d_blkSums);
    if (gridSize.x != 1) {
        computeScanArray(d_blkSums, d_sum_blkSums, gridSize.x, blkSize, blockSizeCTA2);
    }
    sumPrefixBlkKernel<<<gridSize, blkSize>>>(d_out, n, d_sum_blkSums);

    CHECK(cudaFree(d_sum_blkSums));
    CHECK(cudaFree(d_blkSums));
}

__global__ void scatterKernel(uint32_t* src, int n, uint32_t* dst, uint32_t* histScan, int bit, int nBins, uint32_t* count) {
    extern __shared__ uint32_t start[];
    uint32_t first = 2 * CTA_SIZE * blockDim.x * blockIdx.x;
    for (int i = threadIdx.x; i < nBins; i += blockDim.x) {
        start[CONFLICT_FREE_OFFSET(i)] = histScan[blockIdx.x * nBins + i];
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < 2 * CTA_SIZE * blockDim.x; i += blockDim.x) {
        if (first + i < n) {
            uint32_t val = src[first + i];
            uint32_t st = start[CONFLICT_FREE_OFFSET(getBin(val, bit, nBins))];
            uint32_t equalsBefore = count[first + i];
            uint32_t pos = st + equalsBefore - 1;
            dst[pos] = val;
        }
    }
}

__global__ void sortLocalKernel(uint32_t* src, int n, int bit, int nBins, int k, uint32_t* count, uint32_t* hist) {
    extern __shared__ uint32_t s[];
    uint32_t* localSrc = s;
    uint32_t* localBin = s + CONFLICT_FREE_OFFSET(2 * CTA_SIZE * blockDim.x);
    uint32_t* localScan = localBin + CONFLICT_FREE_OFFSET(2 * blockDim.x);

    int ai = threadIdx.x;
    int bi = threadIdx.x + blockDim.x;
    
    for (int i = threadIdx.x; i < 2 * CTA_SIZE * blockDim.x; i += blockDim.x) {
        int pos = (2 * CTA_SIZE * blockDim.x) * blockIdx.x + i;
        localSrc[CONFLICT_FREE_OFFSET(i)] = pos < n ? src[pos] : UINT_MAX;
    }
    __syncthreads();

    uint32_t tempA[CTA_SIZE], tempB[CTA_SIZE];
    for (int blockBit = bit; blockBit < bit + k; ++blockBit) {
        uint32_t valA = 0, valB = 0;
        # pragma unroll
        for (int i = 0; i < CTA_SIZE; ++i) {
            tempA[i] = localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)]; 
            valA += (tempA[i] >> blockBit & 1);
            tempB[i] = localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)]; 
            valB += (tempB[i] >> blockBit & 1);
        }

        // compute scan
        localScan[CONFLICT_FREE_OFFSET(ai)] = valA;
        localScan[CONFLICT_FREE_OFFSET(bi)] = valB;
        __syncthreads();

        // reduction phase
        # pragma unroll
        for (int stride = 1, d = BLOCKDIM_SORTLOCAL; stride <= BLOCKDIM_SORTLOCAL; stride <<= 1, d >>= 1) {
            if (threadIdx.x < d) {
                int cur = 2 * stride * (threadIdx.x + 1) - 1;
                int prev = cur - stride;
                localScan[CONFLICT_FREE_OFFSET(cur)] += localScan[CONFLICT_FREE_OFFSET(prev)];
            }
            __syncthreads();
        }
        // post-reduction phase
        # pragma unroll
        for (int stride = BLOCKDIM_SORTLOCAL >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1) {
            if (threadIdx.x < d - 1) {
                int prev = 2 * stride * (threadIdx.x + 1) - 1;
                int cur = prev + stride;
                localScan[CONFLICT_FREE_OFFSET(cur)] += localScan[CONFLICT_FREE_OFFSET(prev)];
            }
            __syncthreads();
        }
        
        // scatter
        int n0 = 2 * CTA_SIZE * blockDim.x - localScan[CONFLICT_FREE_OFFSET(2 * blockDim.x - 1)];
        
        valA = localScan[CONFLICT_FREE_OFFSET(ai)];
        valB = localScan[CONFLICT_FREE_OFFSET(bi)];
        # pragma unroll
        for (int i = CTA_SIZE - 1; i >= 0; --i) {
            if (tempA[i] >> blockBit & 1)
                localSrc[CONFLICT_FREE_OFFSET(n0 + valA - 1)] = tempA[i];
            else
                localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i - valA)] = tempA[i];
            valA -= (tempA[i] >> blockBit & 1);
            
            if (tempB[i] >> blockBit & 1)
                localSrc[CONFLICT_FREE_OFFSET(n0 + valB - 1)] = tempB[i];
            else
                localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i - valB)] = tempB[i];
            valB -= (tempB[i] >> blockBit & 1);
        }
        
        __syncthreads();
    }
    
    // -------------------------------------------------------------------
    // countEqualsBefore
    uint32_t countA[CTA_SIZE], countB[CTA_SIZE];
    #pragma unroll
    for (int i = 0; i < CTA_SIZE; ++i) {
        tempA[i] = getBin(localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)], bit, nBins); 
        tempB[i] = getBin(localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)], bit, nBins); 
        countA[i] = countB[i] = 1;
        if (i) {
            if (tempA[i] == tempA[i - 1])
                countA[i] += countA[i - 1];
            if (tempB[i] == tempB[i - 1])
                countB[i] += countB[i - 1];
        }
    }

    localScan[CONFLICT_FREE_OFFSET(ai)] = countA[CTA_SIZE - 1];
    localScan[CONFLICT_FREE_OFFSET(bi)] = countB[CTA_SIZE - 1];
    localBin[CONFLICT_FREE_OFFSET(ai)] = tempA[CTA_SIZE - 1];
    localBin[CONFLICT_FREE_OFFSET(bi)] = tempB[CTA_SIZE - 1];
    __syncthreads();

    // reduction phase
    # pragma unroll
    for (int stride = 1, d = BLOCKDIM_SORTLOCAL; stride <= BLOCKDIM_SORTLOCAL; stride <<= 1, d >>= 1) {
        if (threadIdx.x < d) {
            int cur = 2 * stride * (threadIdx.x + 1) - 1;
            int prev = cur - stride;
            cur = CONFLICT_FREE_OFFSET(cur);
            prev = CONFLICT_FREE_OFFSET(prev);
            if (localBin[cur] == localBin[prev])
                localScan[cur] += localScan[prev];
        }
        __syncthreads();
    }
    // post-reduction phase
    # pragma unroll
    for (int stride = BLOCKDIM_SORTLOCAL >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1) {
        if (threadIdx.x < d - 1) {
            int prev = 2 * stride * (threadIdx.x + 1) - 1;
            int cur = prev + stride;
            cur = CONFLICT_FREE_OFFSET(cur);
            prev = CONFLICT_FREE_OFFSET(prev);
            if (localBin[cur] == localBin[prev])
                localScan[cur] += localScan[prev];
        }
        __syncthreads();
    }

    uint32_t lastBinA = localBin[CONFLICT_FREE_OFFSET(ai - 1)];
    uint32_t lastBinB = localBin[CONFLICT_FREE_OFFSET(bi - 1)];
    uint32_t lastScanA = localScan[CONFLICT_FREE_OFFSET(ai - 1)];
    uint32_t lastScanB = localScan[CONFLICT_FREE_OFFSET(bi - 1)];
    __syncthreads();
    
    #pragma unroll
    for (int i = 0; i < CTA_SIZE; ++i) {
        if (threadIdx.x && tempA[i] == lastBinA)
            countA[i] += lastScanA;
        
        if (tempB[i] == lastBinB)
            countB[i] += lastScanB;
        
        localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)] = countA[i];
        localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)] = countB[i];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 2 * CTA_SIZE * blockDim.x; i += blockDim.x) {
        int pos = 2 * CTA_SIZE * blockDim.x * blockIdx.x + i;
        if (pos < n) {
            count[pos] = localScan[CONFLICT_FREE_OFFSET(i)];
            src[pos] = localSrc[CONFLICT_FREE_OFFSET(i)];
        }
    }
    
    // -------------------------------------------
    // compute hist
    uint32_t* s_hist = localScan + CONFLICT_FREE_OFFSET(2 * CTA_SIZE * blockDim.x);
    for (int idx = threadIdx.x; idx < nBins; idx += blockDim.x)
        s_hist[CONFLICT_FREE_OFFSET(idx)] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < 2 * CTA_SIZE * blockDim.x; i += blockDim.x) {
        int pos = 2 * CTA_SIZE * blockDim.x * blockIdx.x + i;
        if (pos < n) {
            uint32_t thisBin = getBin(localSrc[CONFLICT_FREE_OFFSET(i)], bit, nBins);
          if (pos == n - 1 || i == 2 * CTA_SIZE * blockDim.x - 1 || thisBin != getBin(localSrc[CONFLICT_FREE_OFFSET(i + 1)], bit, nBins))
              s_hist[CONFLICT_FREE_OFFSET(thisBin)] = localScan[CONFLICT_FREE_OFFSET(i)];
        }
    }
    __syncthreads();
    
    for (int digit = threadIdx.x; digit < nBins; digit += blockDim.x)
        hist[blockIdx.x * nBins + digit] = s_hist[CONFLICT_FREE_OFFSET(digit)];
}

__global__ void transpose(uint32_t *iMatrix, uint32_t *oMatrix, int rows, int cols)
{
    __shared__ int s_blkData[32][33];
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


void sort(const uint32_t * in, int n, uint32_t * out, int k, int blkSize) {
    int nBins = 1 << k;
    uint32_t * d_src;
    uint32_t * d_dst;
    uint32_t * d_hist;
    uint32_t * d_histScan;
    uint32_t * d_count;
    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_count, n * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_src, in, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_dst, n * sizeof(uint32_t)));

    // Compute block and grid size for scan and scatter phase
    dim3 blockSize(blkSize);
    dim3 blockSize2(blkSize / 2);
    dim3 blockSizeCTA(blkSize / CTA_SIZE);
    dim3 blockSizeCTA2(blkSize / CTA_SIZE / 2);
    dim3 gridSize((n - 1) / blockSize.x + 1);
    dim3 blockSizeTranspose(32, 32);
    dim3 gridSizeTransposeHist((gridSize.x - 1) / blockSizeTranspose.x + 1, (nBins - 1) / blockSizeTranspose.x + 1);
    dim3 gridSizeTransposeHistScan((nBins - 1) / blockSizeTranspose.x + 1, (gridSize.x - 1) / blockSizeTranspose.x + 1);
    
    int histSize = nBins * gridSize.x;
    CHECK(cudaMalloc(&d_hist, 2 * histSize * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_histScan, 2 * histSize * sizeof(uint32_t)));
    dim3 gridSizeScan((histSize - 1) / blockSize.x + 1);

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += k) {
        sortLocalKernel<<<gridSize, blockSizeCTA2, CONFLICT_FREE_OFFSET((4 * CTA_SIZE + 2) * blockSizeCTA2.x + nBins) * sizeof(uint32_t)>>>
            (d_src, n, bit, nBins, k, d_count, d_hist + histSize);
//         printArr("src after", d_src, n);
//         printArr("count", d_count, n);
        
        transpose<<<gridSizeTransposeHist, blockSizeTranspose>>>
          (d_hist + histSize, d_hist, gridSize.x, nBins);
        
        // compute hist scan
        computeScanArray(d_hist, d_histScan + histSize, histSize, blockSize, blockSizeCTA2);
        reduceKernel<<<gridSizeScan, blockSize>>>
            (d_hist, histSize, d_histScan + histSize);
        
        // scatter
        transpose<<<gridSizeTransposeHistScan, blockSizeTranspose>>>
          (d_histScan + histSize, d_histScan, nBins, gridSize.x);
        scatterKernel<<<gridSize, blockSizeCTA2, CONFLICT_FREE_OFFSET(nBins) * sizeof(uint32_t)>>>
            (d_src, n, d_dst, d_histScan, bit, nBins, d_count);
        uint32_t * tmp = d_src; d_src = d_dst; d_dst = tmp;
    }

    CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_histScan));
}
