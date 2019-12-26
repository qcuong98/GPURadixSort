CC = nvcc
PROF = nvprof
K = 8
BLKSIZE1 = 1024
BLKSIZE2 = 1024
BLKSIZE3 = 1024

all: build run clean
build:
	@${CC} main.cu sort.cu thrust.cu utils.cu -o main
run:
	@./main ${K} ${BLKSIZE1} ${BLKSIZE2} ${BLKSIZE3}
clean:
	@rm ./main

benchmark:
	@${CC} benchmark.cu sort.cu thrust.cu utils.cu -o main
	@${PROF} ./main ${K} ${BLKSIZE1} ${BLKSIZE2} ${BLKSIZE3}
	@rm ./main