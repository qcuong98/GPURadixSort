CC = nvcc
PROF = nvprof
K = 8
BLKSIZE1 = 1024
BLKSIZE2 = 1024

all: build run clean
benchmark: build prof clean
build:
	@${CC} main.cu sort.cu thrust.cu utils.cu -o main
run:
	@./main ${K} ${BLKSIZE1} ${BLKSIZE2}
prof:
	@${PROF} ./main ${K} ${BLKSIZE1} ${BLKSIZE2}
clean:
	@rm ./main
