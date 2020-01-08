CC = nvcc
PROF = nvprof
K = 8
BLKSIZE = 256
CTA = 4
STREAMS = 16
N = 16777217

all: build run clean
build:
	@${CC} main.cu sort.cu thrust.cu utils.cu -DK_BITS=${K} -DBLOCKSIZE=${BLKSIZE} -DCTA_SIZE=${CTA} -DN_STREAMS=${STREAMS} -O2 -o main
run:
	@./main ${N}
clean:
	@rm ./main

benchmark:
	@${CC} benchmark.cu sort.cu thrust.cu utils.cu -DK_BITS=${K} -DBLOCKSIZE=${BLKSIZE} -DCTA_SIZE=${CTA} -DN_STREAMS=${STREAMS} -o main
	@${PROF} ./main ${N}
	@rm ./main
