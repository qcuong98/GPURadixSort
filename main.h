#ifndef _MAIN_H_
#define _MAIN_H_

#include <stdint.h>
#include <stdio.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
}                                                                              \
}

void printDeviceInfo();
void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n);
void sort(const uint32_t * in, int n, uint32_t * out);
void sortByThrust(const uint32_t * in, int n, uint32_t * out);

#endif
