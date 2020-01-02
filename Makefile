CC = nvcc
PROF = nvprof
K = 4
BLKSIZE = 256

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