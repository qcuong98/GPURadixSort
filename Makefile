CC = nvcc
PROF = nvprof
K = 8
BLKSIZE = 128

all: build run clean
build:
	@${CC} main.cu sort.cu thrust.cu utils.cu -DK_BITS=${K} -DBLOCKSIZE=${BLKSIZE} -O2 -o main
run:
	@./main
clean:
	@rm ./main

benchmark:
	@${CC} benchmark.cu sort.cu thrust.cu utils.cu -DK_BITS=${K} -DBLOCKSIZE=${BLKSIZE} -o main
	@${PROF} ./main
	@rm ./main
