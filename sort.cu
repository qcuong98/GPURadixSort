#include "main.h"
#include "timer.h"

void sort(const uint32_t * in, int n, uint32_t * out, int k, int * blockSizes) {
    GpuTimer timer;
    timer.Start();

    int nBins = 1 << k; // 2^k
    int * hist = (int *)malloc(nBins * sizeof(int));
    int * histScan = (int *)malloc(nBins * sizeof(int));

    // In each counting sort, we sort data in "src" and write result to "dst"
    // Then, we swap these 2 pointers and go to the next counting sort
    // At first, we assign "src = in" and "dest = out"
    // However, the data pointed by "in" is read-only 
    // --> we create a copy of this data and assign "src" to the address of this copy
    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;

    // Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
    // (Each digit consists of k bits)
	// In each loop, sort elements according to the current digit 
	// (using STABLE counting sort)
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += k)
    {
        // Compute "hist" of the current digit
        memset(hist, 0, nBins * sizeof(int));
        for (int i = 0; i < n; ++i) {
            int bin = (src[i] >> bit) & (nBins - 1);
            ++hist[bin];
        }

        // Scan "hist" (exclusively) and save the result to "histScan"
        histScan[0] = 0;
        for (int i = 1; i < nBins; ++i)
            histScan[i] = histScan[i - 1] + hist[i - 1];

        // From "histScan", scatter elements in "src" to correct locations in "dst"
        for (int i = 0; i < n; ++i) {
            int bin = (src[i] >> bit) & (nBins - 1);
            dst[histScan[bin]] = src[i];
            ++histScan[bin];
        }
    	
        // Swap "src" and "dst"
        uint32_t * tmp = src;
        src = dst;
        dst = tmp;
    }

    // Copy result to "out"
    memcpy(out, src, n * sizeof(uint32_t));

    // Free memories
    free(hist);
    free(histScan);
    free(originalSrc);

    timer.Stop();
    printf("Our Time: %.3f ms\n", timer.Elapsed());
}