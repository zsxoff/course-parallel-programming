#include <cublas_v2.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);
  printf("n = %d\n", n);

  int n2b = n * sizeof(int);
  int n2 = n;

  // Выделение памяти на хосте.
  int *a = (int *)calloc(n2, sizeof(int));
  int *b = (int *)calloc(n2, sizeof(int));
  int *c = (int *)calloc(n2, sizeof(int));

  // Инициализация массивов.
  a[0] = 0;
  a[1] = 1;
  a[2] = 2;
  a[3] = 3;
  a[4] = 4;
  a[5] = 5;
  a[6] = 6;
  a[7] = 7;

  b[0] = 0;
  b[1] = 1;
  b[2] = 2;
  b[3] = 3;
  b[4] = 4;
  b[5] = 5;
  b[6] = 6;
  b[7] = 7;

  for (int i = 8; i < n; ++i) {
    a[i] = 1;
    b[i] = 1;
  }

  // Выделение памяти на устройстве.
  int *adev = NULL;
  cudaError_t cuerr = cudaMalloc((void **)&adev, n2b);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot allocate device array for a: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  int *bdev = NULL;
  cuerr = cudaMalloc((void **)&bdev, n2b);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot allocate device array for b: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  int *cdev = NULL;
  cuerr = cudaMalloc((void **)&cdev, n2b);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot allocate device array for c: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  // Создание обработчиков событий.
  cudaEvent_t start, stop;
  float gpuTime = 0.0f;

  cuerr = cudaEventCreate(&start);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot create CUDA start event: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  cuerr = cudaEventCreate(&stop);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot create CUDA end event: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  // Копирование данных с хоста на GPU.
  cuerr = cudaMemcpy(adev, a, n2b, cudaMemcpyHostToDevice);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot copy a array from host to device: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  cuerr = cudaMemcpy(bdev, b, n2b, cudaMemcpyHostToDevice);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot copy b array from host to device: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  // Установка точки старта.
  cuerr = cudaEventRecord(start, 0);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot record CUDA event: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  // Запуск ядра.
  int BLOCK_NUMBER = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  kernel<<<BLOCK_NUMBER, BLOCK_SIZE>>>(cdev, adev, bdev, n);

  cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  // Синхронизация устройств.
  cuerr = cudaDeviceSynchronize();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  // Установка точки окончания.
  cuerr = cudaEventRecord(stop, 0);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  // Копирование результата на хост.
  cuerr = cudaMemcpy(c, cdev, n2b, cudaMemcpyDeviceToHost);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  // Расчет времени.
  cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
  printf("time spent executing %s: %.9f seconds\n", "kernel", gpuTime / 1000);

  for (int i = 0; i < 10; ++i) {
    printf(c[i]);
  }

  /* Очистка памяти. */

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(adev);
  cudaFree(bdev);
  cudaFree(cdev);

  free(a);
  free(b);
  free(c);

  return 0;
}
