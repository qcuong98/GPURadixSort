#include "main.h"

int main(int argc, char ** argv) {
    // SET UP INPUT SIZE
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // SET UP K
    int k = K_BITS; // atoi(argv[1]);
    printf("\nNum bits per digit: %d\n", k);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Our result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = rand();
    // printArray(in, n);

    // DETERMINE BLOCK SIZES
    int blockSize = BLOCKSIZE; // atoi(argv[2]);
    printf("\nBlock size for all kernels: %d\n", blockSize);

    sort(in, n, out);

    // FREE MEMORIES
    free(in);
    free(out);
    
    return EXIT_SUCCESS;
}
