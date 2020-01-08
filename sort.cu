#include "main.h"
#include <bits/stdc++.h>

#define CTA_SIZE 4
#define N_STREAMS 16

#define ELEMENTS_PER_BLOCK (2 * CTA_SIZE * BLOCKSIZE)
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) + ((n) >> LOG_NUM_BANKS))

__device__ uint32_t getBin(uint32_t val, uint32_t bit, uint32_t nBins) {
    return (val >> bit) & (nBins - 1);
}

__global__ void scanBlkKernel(uint32_t * src, int n, uint32_t * out, uint32_t * blkSums) {
    extern __shared__ uint32_t s[];
    uint32_t* localScan = s;
    uint32_t* localScanCTA = localScan + CONFLICT_FREE_OFFSET(2 * CTA_SIZE * blockDim.x);

    int ai = threadIdx.x;
    int bi = threadIdx.x + blockDim.x;
    
    uint32_t first = ELEMENTS_PER_BLOCK * blockIdx.x;
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
        int pos = first + i;
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
    for (int stride = 1, d = BLOCKSIZE; stride <= BLOCKSIZE; stride <<= 1, d >>= 1) {
        if (threadIdx.x < d) {
            int cur = 2 * stride * (threadIdx.x + 1) - 1;
            int prev = cur - stride;
            localScanCTA[CONFLICT_FREE_OFFSET(cur)] += localScanCTA[CONFLICT_FREE_OFFSET(prev)];
        }
        __syncthreads();
    }
    // post-reduction phase
    # pragma unroll
    for (int stride = BLOCKSIZE >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1) {
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
    
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
        if (first + i < n)
            out[first + i] = localScan[CONFLICT_FREE_OFFSET(i)];
    }
    
    if (threadIdx.x == blockDim.x - 1) {
        blkSums[blockIdx.x] = tempB[CTA_SIZE - 1];
    }
}

__global__ void sumPrefixBlkKernel(uint32_t * out, int n, uint32_t * blkSums) {
    uint32_t lastBlockSum = blockIdx.x > 0 ? blkSums[blockIdx.x - 1] : 0;
    uint32_t first = ELEMENTS_PER_BLOCK * blockIdx.x;
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
        if (first + i < n)
            out[first + i] += lastBlockSum;
    }
}

__global__ void reduceKernel(uint32_t * in, int n, uint32_t * out) {
    int id_in = blockDim.x * blockIdx.x + threadIdx.x;
    if (id_in < n)
        out[id_in] -= in[id_in];
}

void computeScanArray(uint32_t* d_in, uint32_t* d_out, int n, dim3 elementsPerBlock, dim3 blockSize) {
    dim3 gridSize((n - 1) / elementsPerBlock.x + 1);

    uint32_t * d_blkSums;
    CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(uint32_t)));
    uint32_t * d_sum_blkSums;
    CHECK(cudaMalloc(&d_sum_blkSums, gridSize.x * sizeof(uint32_t)));

    scanBlkKernel<<<gridSize, blockSize, CONFLICT_FREE_OFFSET((2 * CTA_SIZE + 2) * blockSize.x) * sizeof(uint32_t)>>>
        (d_in, n, d_out, d_blkSums);
    if (gridSize.x != 1) {
        computeScanArray(d_blkSums, d_sum_blkSums, gridSize.x, elementsPerBlock, blockSize);
    }
    sumPrefixBlkKernel<<<gridSize, blockSize>>>(d_out, n, d_sum_blkSums);

    CHECK(cudaFree(d_sum_blkSums));
    CHECK(cudaFree(d_blkSums));
}

__global__ void scatterKernel(uint32_t* src, int n, uint32_t* dst, uint32_t* histScan, int bit, int nBins, uint32_t* count) {
    extern __shared__ uint32_t start[];
    uint32_t first = ELEMENTS_PER_BLOCK * blockIdx.x;
    for (int i = threadIdx.x; i < nBins; i += blockDim.x) {
        start[CONFLICT_FREE_OFFSET(i)] = histScan[blockIdx.x * nBins + i];
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
        if (first + i < n) {
            uint32_t val = src[first + i];
            uint32_t st = start[CONFLICT_FREE_OFFSET(getBin(val, bit, nBins))];
            uint32_t equalsBefore = count[first + i];
            uint32_t pos = st + equalsBefore - 1;
            dst[pos] = val;
        }
    }
}

__global__ void sortLocalKernel(uint32_t* src, int n, int bit, int nBins, int k, uint32_t* count, uint32_t* hist, int start_pos = 0) {
    extern __shared__ uint32_t s[];
    uint32_t* localSrc = s;
    uint32_t* localBin = s + CONFLICT_FREE_OFFSET(ELEMENTS_PER_BLOCK);
    uint32_t* localScan = localBin + CONFLICT_FREE_OFFSET(2 * BLOCKSIZE);
    uint32_t* s_hist = localScan + CONFLICT_FREE_OFFSET(ELEMENTS_PER_BLOCK);

    int ai = threadIdx.x;
    int bi = threadIdx.x + blockDim.x;
    uint32_t first = ELEMENTS_PER_BLOCK * blockIdx.x;
    
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
        int pos = first + i;
        localSrc[CONFLICT_FREE_OFFSET(i)] = pos < n ? src[pos] : UINT_MAX;
    }
    __syncthreads();

    // radix sort with k = 1
    uint32_t tempA[CTA_SIZE], tempB[CTA_SIZE];
    #pragma unroll
    for (int b = 0; b < K_BITS; ++b) {
        int blockBit = bit + b;
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
        for (int stride = 1, d = BLOCKSIZE; stride <= BLOCKSIZE; stride <<= 1, d >>= 1) {
            if (threadIdx.x < d) {
                int cur = 2 * stride * (threadIdx.x + 1) - 1;
                int prev = cur - stride;
                localScan[CONFLICT_FREE_OFFSET(cur)] += localScan[CONFLICT_FREE_OFFSET(prev)];
            }
            __syncthreads();
        }
        // post-reduction phase
        # pragma unroll
        for (int stride = BLOCKSIZE >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1) {
            if (threadIdx.x < d - 1) {
                int prev = 2 * stride * (threadIdx.x + 1) - 1;
                int cur = prev + stride;
                localScan[CONFLICT_FREE_OFFSET(cur)] += localScan[CONFLICT_FREE_OFFSET(prev)];
            }
            __syncthreads();
        }
        
        // scatter
        int n0 = ELEMENTS_PER_BLOCK - localScan[CONFLICT_FREE_OFFSET(2 * blockDim.x - 1)];
        
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
    for (int stride = 1, d = BLOCKSIZE; stride <= BLOCKSIZE; stride <<= 1, d >>= 1) {
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
    for (int stride = BLOCKSIZE >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1) {
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
    uint32_t lastScanA = ai ? localScan[CONFLICT_FREE_OFFSET(ai - 1)] : 0;
    uint32_t lastScanB = localScan[CONFLICT_FREE_OFFSET(bi - 1)];
    __syncthreads();
    
    #pragma unroll
    for (int i = 0; i < CTA_SIZE; ++i) {
        if (tempA[i] == lastBinA)
            countA[i] += lastScanA;
        
        if (tempB[i] == lastBinB)
            countB[i] += lastScanB;
        
        localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)] = countA[i];
        localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)] = countB[i];
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
        int pos = first + i;
        if (pos < n) {
            count[pos] = localScan[CONFLICT_FREE_OFFSET(i)];
            src[pos] = localSrc[CONFLICT_FREE_OFFSET(i)];
        }
    }
    
    // -------------------------------------------
    // compute hist
    for (int idx = threadIdx.x; idx < nBins; idx += blockDim.x)
        s_hist[CONFLICT_FREE_OFFSET(idx)] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
        int pos = first + i;
        if (pos < n) {
            uint32_t thisBin = getBin(localSrc[CONFLICT_FREE_OFFSET(i)], bit, nBins);
          if (pos == n - 1 || i == ELEMENTS_PER_BLOCK - 1 || thisBin != getBin(localSrc[CONFLICT_FREE_OFFSET(i + 1)], bit, nBins))
              s_hist[CONFLICT_FREE_OFFSET(thisBin)] = localScan[CONFLICT_FREE_OFFSET(i)];
        }
    }
    __syncthreads();
    
    first = (blockIdx.x + start_pos) * nBins;
    for (int digit = threadIdx.x; digit < nBins; digit += blockDim.x)
        hist[first + digit] = s_hist[CONFLICT_FREE_OFFSET(digit)];
}

__global__ void transpose(uint32_t *iMatrix, uint32_t *oMatrix, int rows, int cols) {
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

void sort(const uint32_t * in, int n, uint32_t * out) {
    int nBins = 1 << K_BITS;
    uint32_t * d_src;
    uint32_t * d_dst;
    uint32_t * d_hist;
    uint32_t * d_histScan;
    uint32_t * d_count;
    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_count, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_dst, n * sizeof(uint32_t)));

    // Compute block and grid size for scan and scatter phase
    dim3 blockSize(BLOCKSIZE);
    dim3 elementsPerBlock(ELEMENTS_PER_BLOCK);
    dim3 gridSize((n - 1) / elementsPerBlock.x + 1);
    dim3 blockSizeTranspose(32, 32);
    dim3 gridSizeTransposeHist((gridSize.x - 1) / blockSizeTranspose.x + 1, (nBins - 1) / blockSizeTranspose.x + 1);
    dim3 gridSizeTransposeHistScan((nBins - 1) / blockSizeTranspose.x + 1, (gridSize.x - 1) / blockSizeTranspose.x + 1);
    
    int histSize = nBins * gridSize.x;
    CHECK(cudaMalloc(&d_hist, 2 * histSize * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_histScan, 2 * histSize * sizeof(uint32_t)));
    dim3 gridSizeScan((histSize - 1) / blockSize.x + 1);

    cudaStream_t *streams = (cudaStream_t *) malloc(N_STREAMS * sizeof(cudaStream_t));    
    for (int i = 0; i < N_STREAMS; ++i) {
        CHECK(cudaStreamCreate(&streams[i]));
    }
    int len = (gridSize.x - 1) / N_STREAMS + 1;
    for (int i = 0; i < N_STREAMS; ++i) {
        int cur_pos = i * len * elementsPerBlock.x;
        if (cur_pos >= n)
            break;
        int cur_len = min(len * elementsPerBlock.x, n - i * len * elementsPerBlock.x);
        dim3 cur_gridSize((cur_len - 1) / elementsPerBlock.x + 1);
        CHECK(cudaMemcpyAsync(d_src + cur_pos, in + cur_pos, cur_len * sizeof(uint32_t), 
                                            cudaMemcpyHostToDevice, streams[i]));
        sortLocalKernel<<<cur_gridSize, blockSize, CONFLICT_FREE_OFFSET((4 * CTA_SIZE + 2) * blockSize.x + nBins) * sizeof(uint32_t), streams[i]>>>
            (d_src + cur_pos, cur_len, 0, nBins, K_BITS, d_count + cur_pos, d_hist + histSize, i * len);
    }

    for (int bit = 0; bit < 32; bit += K_BITS) {
        if (bit) {
            sortLocalKernel<<<gridSize, blockSize, CONFLICT_FREE_OFFSET((4 * CTA_SIZE + 2) * BLOCKSIZE + nBins) * sizeof(uint32_t)>>>
                (d_src, n, bit, nBins, K_BITS, d_count, d_hist + histSize);
        }
        
        transpose<<<gridSizeTransposeHist, blockSizeTranspose>>>
          (d_hist + histSize, d_hist, gridSize.x, nBins);

        // compute hist scan
        computeScanArray(d_hist, d_histScan + histSize, histSize, elementsPerBlock, blockSize);
        reduceKernel<<<gridSizeScan, blockSize>>>
            (d_hist, histSize, d_histScan + histSize);
        
        transpose<<<gridSizeTransposeHistScan, blockSizeTranspose>>>
          (d_histScan + histSize, d_histScan, nBins, gridSize.x);
        
        // scatter
        scatterKernel<<<gridSize, blockSize, CONFLICT_FREE_OFFSET(nBins) * sizeof(uint32_t)>>>
            (d_src, n, d_dst, d_histScan, bit, nBins, d_count);
        uint32_t * tmp = d_src; d_src = d_dst; d_dst = tmp;
    }

    CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_histScan));
}
