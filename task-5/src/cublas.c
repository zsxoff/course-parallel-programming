#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <time.h>

#define N 3

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n).
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m,
                   const int k, const int n) {
  int lda = m;
  int ldb = k;
  int ldc = m;

  const float alf = 1;
  const float bet = 0;

  const float *alpha = &alf;
  const float *beta = &bet;

  // Create a handle for CUBLAS.
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Do the actual multiplication.
  cublasSgemm(handle,      // handle
              CUBLAS_OP_N, // transa
              CUBLAS_OP_N, // transb
              m,           // m
              n,           // n
              k,           // k
              alpha,       // *alpha
              A,           // *A
              lda,         // lda
              B,           // *B
              ldb,         // ldb
              beta,        // *beta
              C,           // *C
              ldc          // ldc
  );

  // Destroy the handle.
  cublasDestroy(handle);
}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU.
void GPU_fill_rand(float *A, const int nr_rows_A, const int nr_cols_A) {
  // Create a pseudo-random number generator.
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_MTGP32);

  // Set the seed for the random number generator using the system clock.
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

  // Fill the array with random numbers on the device.
  curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

int main() {
  // Allocate arrays on CPU.
  float a[N][N];
  float b[N][N];

  // srand(time(NULL));

  int i;
  int j;

  a[0][0] = 1;
  a[1][0] = 0;
  a[2][0] = 4;

  a[0][1] = 2;
  a[1][1] = 3;
  a[2][1] = 1;

  a[0][2] = 3;
  a[1][2] = 0;
  a[2][2] = 2;

  b[0][0] = 1;
  b[1][0] = 1;
  b[2][0] = 2;

  b[0][1] = 2;
  b[1][1] = 0;
  b[2][1] = 3;

  b[0][2] = 3;
  b[1][2] = 4;
  b[2][2] = 2;

  const int nr_rows_A = N, nr_cols_A = N;
  const int nr_rows_B = N, nr_cols_B = N;
  const int nr_rows_C = N, nr_cols_C = N;

  float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
  float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
  float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

  for (i = 0; i < nr_rows_A; ++i) {
    for (j = 0; j < nr_cols_A; ++j) {
      h_A[i * nr_cols_A + j] = a[i][j];
      h_B[i * nr_cols_A + j] = b[i][j];
    }
  }

  // Allocate arrays on GPU.
  float *d_A;
  float *d_B;
  float *d_C;

  cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(float));
  cudaMalloc(&d_B, nr_rows_B * nr_cols_B * sizeof(float));
  cudaMalloc(&d_C, nr_rows_C * nr_cols_C * sizeof(float));

  // Fill the arrays A and B on GPU with random numbers.
  // GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
  // GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);

  // Optionally we can copy the data back on CPU and print the arrays.
  cudaMemcpy(d_A,                                   //
             h_A,                                   //
             nr_rows_A * nr_cols_A * sizeof(float), //
             cudaMemcpyHostToDevice                 //
  );

  cudaMemcpy(d_B,                                   //
             h_B,                                   //
             nr_rows_B * nr_cols_B * sizeof(float), //
             cudaMemcpyHostToDevice                 //
  );

  // Multiply A and B on GPU.
  clock_t begin = clock();

  gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

  double elapsed_ses = (double)((clock() - begin)) / CLOCKS_PER_SEC;

  printf("Dimension: %d\nTime: %f ms", N, elapsed_ses);

  // Copy (and print) the result on host memory.
  cudaMemcpy(h_C,                                   //
             d_C,                                   //
             nr_rows_C * nr_cols_C * sizeof(float), //
             cudaMemcpyDeviceToHost                 //
  );

  printf("\n");
  printf("\n");
  printf("\n");

  for (i = 0; i < nr_rows_C; ++i) {
    for (j = 0; j < nr_cols_C; ++j) {
      printf("%f ", h_C[j * nr_cols_C + i]);
    }
    printf("\n");
  }
  printf("\n");

  // Free GPU memory.
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free CPU memory.
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
