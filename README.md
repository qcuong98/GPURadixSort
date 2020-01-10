# An Implementation Of Radix Sort In CUDA

# Command Line Usage
## Build
```
make build [K=Value] [BLKSIZE=Value] [CTA=Value] [STREAMS=Value]
```

The best performance can be achieved with BLKSIZE = 256, CTA = 4, STREAMS = 16
## Run
```
make run [N=Value]
```
## Clean
```
make clean
```
## Or you can just
```
make
```
## Example Output:
```
**********GPU info**********
Name: NVIDIA GeForce RTX 2070 SUPER
Compute capability: 7.5
Num SMs: 40
Max num threads per SM: 1024
Max num warps per SM: 32
GMEM: 8GB
SMEM per SM: 65536 byte
SMEM per block: 49152 byte

****************************

Input size: 16777217

Num bits per digit: 8

Block size for all kernels: 256

Our Time: 50.152 ms
Thrust's Time: 42.387 ms
CORRECT :)
```
