#include "main.h"
#include "timer.h"

int main(int argc, char ** argv) {
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    int n = atoi(argv[1]);
    printf("\nInput size: %d\n", n);

    // SET UP K
    int k = K_BITS; // atoi(argv[1]);
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
    int blockSize = BLOCKSIZE; // atoi(argv[2]);
    printf("\nBlock size for all kernels: %d\n", blockSize);

    // SORT BY THRUST
    GpuTimer thrust_timer;
    thrust_timer.Start();
    sortByThrust(in, n, thrustOut);
    thrust_timer.Stop();
    printf("\nThrust's Time: %.3f ms\n", thrust_timer.Elapsed());
    // printArray(thrustOut, n);
    
    // SORT BY OUR IMPLEMENTATION
    GpuTimer our_timer;
    our_timer.Start();
    sort(in, n, out);
    our_timer.Stop();
    printf("Our Time: %.3f ms\n", our_timer.Elapsed());
    // printArray(out, n);

    checkCorrectness(out, thrustOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(thrustOut);
    
    return EXIT_SUCCESS;
}
