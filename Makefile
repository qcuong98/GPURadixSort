CC = nvcc
PROF = nvprof
K = 8
BLKSIZE = 1024

all: build run clean
build:
	@${CC} main.cu sort.cu thrust.cu utils.cu -o main
run:
	@./main ${K} ${BLKSIZE}
clean:
	@rm ./main

benchmark:
	@${CC} benchmark.cu sort.cu thrust.cu utils.cu -o main
	@${PROF} ./main ${K} ${BLKSIZE}
	@rm ./main
