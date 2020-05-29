BLOCK_SIZE = 128 # The size of shared memory block size
NVCC = nvcc
CFLAGS=-g -G -O0 -DBLOCK_SIZE=$(BLOCK_SIZE) -lcublas

all: addGPU

addGPU: src/addGPU.cu src/addGPU.c
	nvcc ${CFLAGS} $< -o $@

clean:
	rm -rf ./addGPU core

.PHONY: addGPU all clean
