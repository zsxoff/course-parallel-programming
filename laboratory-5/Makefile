NVCC = nvcc
CFLAGS=-g -G -O0 -lcublas -lcurand -lcurand_static -lculibos

all: cublas

cublas: src/cublas.c
	nvcc ${CFLAGS} $< -o $@

clean:
	rm -rf ./cublas

.PHONY: cublas all clean
