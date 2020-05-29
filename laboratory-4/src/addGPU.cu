__global__ void addKernel(int *c, int *a, int *b) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  c[index] = a[index] + b[index];
}

#define kernel addKernel
#include "addGPU.c"
