#include "main.h"

int main(int argc, char ** argv) {
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // SET UP K
    int k = atoi(argv[1]);
    printf("\nNum bits per digit: %d\n", k);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Our result
    uint32_t * thrustOut = (uint32_t *)malloc(bytes); // Thrust's result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = rand();
    // printArray(in, n);

    // DETERMINE BLOCK SIZES
    int blockSizes[2] = {512, 512}; // One for histogram, one for scan
    blockSizes[0] = atoi(argv[2]);
    blockSizes[1] = atoi(argv[3]);
    printf("\nHist block size: %d, scan block size: %d\n", 
                                                blockSizes[0], blockSizes[1]);

    // SORT BY THRUST
    sortByThrust(in, n, thrustOut);
    // printArray(thrustOut, n);
    
    // SORT BY OUR IMPLEMENTATION
    sort(in, n, out, k, blockSizes);
    // printArray(out, n);

    checkCorrectness(out, thrustOut, n);

    // FREE MEMORIES 
    free(in);
    free(out);
    free(thrustOut);
    
    return EXIT_SUCCESS;
}