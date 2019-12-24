#include "main.h"
#include "timer.h"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

void sortByThrust(const uint32_t * in, int n, uint32_t * out)
{
    GpuTimer timer;
    timer.Start();

    thrust::device_vector<uint32_t> dv_out(in, in + n);
    thrust::sort(dv_out.begin(), dv_out.end());
    thrust::copy(dv_out.begin(), dv_out.end(), out);

    timer.Stop();
    printf("Thrust's Time: %.3f ms\n", timer.Elapsed());
}