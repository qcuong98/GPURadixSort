#include "main.h"

#define CTA_SIZE 4

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) + ((n) >> LOG_NUM_BANKS))

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

__device__ void countEqualBefore(uint32_t* src, uint32_t* buffer, int bit, int nBins, int ai, int bi) {
    buffer[CONFLICT_FREE_OFFSET(ai)] = 1;
    buffer[CONFLICT_FREE_OFFSET(bi)] = 1;
    __syncthreads();
    // reduction phase
    for (int stride = 1, d = blockDim.x; stride <= blockDim.x; stride <<= 1, d >>= 1) {
        int cur = 2 * stride * (threadIdx.x + 1) - 1;
        int prev = cur - stride;
        cur = CONFLICT_FREE_OFFSET(cur);
        prev = CONFLICT_FREE_OFFSET(prev);
        if (threadIdx.x < d && getBin(src[cur], bit, nBins) == getBin(src[prev], bit, nBins)) {
            buffer[cur] += buffer[prev];
        }
        __syncthreads();
    }
    // post-reduction phase
    for (int stride = blockDim.x >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1) {
        int cur = 2 * stride * (threadIdx.x + 1) + stride - 1;
        int prev = cur - stride;
        cur = CONFLICT_FREE_OFFSET(cur);
        prev = CONFLICT_FREE_OFFSET(prev);
        if (threadIdx.x < d - 1 && getBin(src[cur], bit, nBins) == getBin(src[prev], bit, nBins)) {
            buffer[cur] += buffer[prev];
        }
        __syncthreads();
    }
}

__global__ void scatterKernel(uint32_t* src, int n, uint32_t* dst, uint32_t* histScan, int bit, int nBins, int gridSize) {
    extern __shared__ uint32_t s[];
    uint32_t* localSrc = s;
    uint32_t* localScan = localSrc + CONFLICT_FREE_OFFSET(2 * blockDim.x);

    int id_ai = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    int id_bi = 2 * blockDim.x * blockIdx.x + threadIdx.x + blockDim.x;
    int ai = threadIdx.x;
    int bi = threadIdx.x + blockDim.x;
    localSrc[CONFLICT_FREE_OFFSET(ai)] = id_ai < n ? src[id_ai] : UINT_MAX;
    localSrc[CONFLICT_FREE_OFFSET(bi)] = id_bi < n ? src[id_bi] : UINT_MAX;

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
}

__global__ void scatterKernel2(uint32_t* src, int n, uint32_t* dst, uint32_t* histScan, int bit, int nBins, int gridSize) {
    extern __shared__ uint32_t s[];
    uint32_t* localSrc = s;
    uint32_t* localScan = localSrc + CONFLICT_FREE_OFFSET(2 * CTA_SIZE * blockDim.x);
    uint32_t* localBin = localScan + CONFLICT_FREE_OFFSET(2 * blockDim.x);

    int ai = threadIdx.x;
    int bi = threadIdx.x + blockDim.x;

    for (int i = threadIdx.x; i < 2 * CTA_SIZE * blockDim.x; i += blockDim.x) {
        int pos = (2 * CTA_SIZE * blockDim.x) * blockIdx.x + i;
        localSrc[CONFLICT_FREE_OFFSET(i)] = pos < n ? src[pos] : UINT_MAX;
    }
    __syncthreads();
    
    uint32_t tempA[CTA_SIZE], tempB[CTA_SIZE], countA[CTA_SIZE], countB[CTA_SIZE];
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
    for (int stride = 1, d = blockDim.x; stride <= blockDim.x; stride <<= 1, d >>= 1) {
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
    for (int stride = blockDim.x >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1) {
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

    countA[CTA_SIZE - 1] = localScan[CONFLICT_FREE_OFFSET(ai)];
    countB[CTA_SIZE - 1] = localScan[CONFLICT_FREE_OFFSET(bi)];
    uint32_t lastBinA = localBin[CONFLICT_FREE_OFFSET(ai - 1)];
    uint32_t lastBinB = localBin[CONFLICT_FREE_OFFSET(bi - 1)];
    uint32_t lastScanA = localScan[CONFLICT_FREE_OFFSET(ai - 1)];
    uint32_t lastScanB = localScan[CONFLICT_FREE_OFFSET(bi - 1)];
    for (int i = CTA_SIZE - 2; i >= 0; --i) {
        if (threadIdx.x && tempA[i] == lastBinA)
            countA[i] += lastScanA;
        
        if (tempB[i] == lastBinB)
            countB[i] += lastScanB;
    }
    
    uint32_t pos;
    for (int i = 0; i < CTA_SIZE; ++i) {
        pos = histScan[blockIdx.x + tempA[i] * gridSize] + countA[i] - 1;
        if (pos < n)
            dst[pos] = localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)];
        
        pos = histScan[blockIdx.x + tempB[i] * gridSize] + countB[i] - 1;
        if (pos < n)
            dst[pos] = localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)];
    }
}

__global__ void sortLocalKernel(uint32_t* src, int n, int bit, int k) {
    extern __shared__ uint32_t s[];
    uint32_t* localSrc = s;
    uint32_t* localScan = localSrc + CONFLICT_FREE_OFFSET(2 * CTA_SIZE * blockDim.x);

    int id_ai = CTA_SIZE * (2 * blockDim.x * blockIdx.x + threadIdx.x);
    int id_bi = CTA_SIZE * (2 * blockDim.x * blockIdx.x + threadIdx.x + blockDim.x);
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
        for (int i = 0; i < CTA_SIZE; ++i) {
            tempA[i] = localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)]; 
            valA += (tempA[i] >> blockBit & 1);
        }
        for (int i = 0; i < CTA_SIZE; ++i) {
            tempB[i] = localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)]; 
            valB += (tempB[i] >> blockBit & 1);
        }

        // compute scan
        localScan[CONFLICT_FREE_OFFSET(ai)] = valA;
        localScan[CONFLICT_FREE_OFFSET(bi)] = valB;
        __syncthreads();

        // reduction phase
        for (int stride = 1, d = blockDim.x; stride <= blockDim.x; stride <<= 1, d >>= 1) {
            if (threadIdx.x < d) {
                int cur = 2 * stride * (threadIdx.x + 1) - 1;
                int prev = cur - stride;
                localScan[CONFLICT_FREE_OFFSET(cur)] += localScan[CONFLICT_FREE_OFFSET(prev)];
            }
            __syncthreads();
        }
        // post-reduction phase
        for (int stride = blockDim.x >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1) {
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
        for (int i = CTA_SIZE - 1; i >= 0; --i) {
            if (tempA[i] >> blockBit & 1)
                localSrc[CONFLICT_FREE_OFFSET(n0 + valA - 1)] = tempA[i];
            else
                localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i - valA)] = tempA[i];
            valA -= (tempA[i] >> blockBit & 1);
        }
        valB = localScan[CONFLICT_FREE_OFFSET(bi)];
        for (int i = CTA_SIZE - 1; i >= 0; --i) {
            if (tempB[i] >> blockBit & 1)
                localSrc[CONFLICT_FREE_OFFSET(n0 + valB - 1)] = tempB[i];
            else
                localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i - valB)] = tempB[i];
            valB -= (tempB[i] >> blockBit & 1);
        }
        
        __syncthreads();
    }

    for (int i = 0; i < CTA_SIZE; ++i)
        if (id_ai + i < n)
            src[id_ai + i] = localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)];
    for (int i = 0; i < CTA_SIZE; ++i)
        if (id_bi + i < n)
            src[id_bi + i] = localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)];
}

void sort(const uint32_t * in, int n, uint32_t * out, int k, int blkSize) {
    int nBins = 1 << k;
    uint32_t * d_src;
    uint32_t * d_dst;
    uint32_t * d_hist;
    uint32_t * d_histScan;
    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_src, in, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_dst, n * sizeof(uint32_t)));

    // Compute block and grid size for scan and scatter phase
    dim3 blockSize(blkSize);
    dim3 blockSize2(blkSize / 2);
    dim3 blockSizeCTA(blkSize / CTA_SIZE);
    dim3 blockSizeCTA2(blkSize / CTA_SIZE / 2);
    dim3 gridSize((n - 1) / blockSize.x + 1);

    int histSize = nBins * gridSize.x;
    CHECK(cudaMalloc(&d_hist, histSize * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_histScan, histSize * sizeof(uint32_t)));
    dim3 gridSizeScan((histSize - 1) / blockSize.x + 1);

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += k) {
        // compute hist
        computeHistKernel<<<gridSize, blockSize, nBins * sizeof(uint32_t)>>>
            (d_src, n, d_hist, nBins, bit, gridSize.x);
        
        // compute hist scan
        computeScanArray(d_hist, d_histScan, histSize, blockSize);
        reduceKernel<<<gridSizeScan, blockSize>>>
            (d_hist, histSize, d_histScan);
        
        // scatter
        sortLocalKernel<<<gridSize, blockSizeCTA2, CONFLICT_FREE_OFFSET((2 * CTA_SIZE + 2) * blockSizeCTA2.x) * sizeof(uint32_t)>>>
            (d_src, n, bit, k);
        scatterKernel2<<<gridSize, blockSizeCTA2, CONFLICT_FREE_OFFSET((2 * CTA_SIZE + 4) * blockSizeCTA2.x) * sizeof(uint32_t)>>>
//         scatterKernel<<<gridSize, blockSize2, CONFLICT_FREE_OFFSET(4 * blockSize2.x) * sizeof(uint32_t)>>>
            (d_src, n, d_dst, d_histScan, bit, nBins, gridSize.x);
        uint32_t * tmp = d_src; d_src = d_dst; d_dst = tmp;
    }

    CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_histScan));
}
