#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NMAX 1000000
#define OMP_THREADS 4

static double arr_sum_default(const double *arr, const int size) {
  double sum = 0.0;
  int i;

  for (i = 0; i < size; ++i) {
    sum += arr[i];
  }

  return sum;
}

static double arr_sum_reduction(const double *arr, const int size) {
  double sum = 0.0;
  int i;

#pragma omp parallel for reduction(+ : sum) shared(arr) default(none)
  for (i = 0; i < size; ++i) {
    sum += arr[i];
  }

  return sum;
}

static double arr_sum_ordered(const double *arr, const int size) {
  double sum = 0.0;
  int i;

#pragma omp parallel for ordered shared(arr, sum) default(none)
  for (i = 0; i < size; ++i) {

#pragma omp ordered
    { sum += arr[i]; }
  }

  return sum;
}

static double arr_sum_critical(const double *arr, const int size) {
  double sum = 0.0;
  int i;

#pragma omp parallel for shared(arr, sum) default(none)
  for (i = 0; i < size; ++i) {

#pragma omp critical
    { sum += arr[i]; }
  }

  return sum;
}

static double arr_sum_atomic(const double *arr, const int size) {
  double sum = 0.0;
  int i;

#pragma omp parallel for shared(arr, sum) default(none)
  for (i = 0; i < size; ++i) {

#pragma omp atomic
    sum += arr[i];
  }

  return sum;
}

int main(int argc, char *argv[]) {
  omp_set_num_threads(OMP_THREADS);

  const int N = NMAX;
  double sum = 0;
  int i;

  double *a = malloc(sizeof(double) * N);

  // Fill array.

  /*
  // Manual init.
  // ! WARNING ! Change global NMAX before init.
   a[0] = 1;
   a[1] = 2;
   a[2] = 3;
   a[3] = 4;
   a[4] = 5;
   a[5] = 6;
   a[6] = 7;
   a[7] = 8;
   a[8] = 9;
   a[9] = 0;
   */

  for (i = 0; i < NMAX; ++i) {
    a[i] = 1.0;
  }

  double start_time;
  double end_time;

  printf("OpenMP Threads: %d\n", OMP_THREADS);
  printf("Array size: %d", N);

  // Method: default.
  sum = 0.0;
  start_time = omp_get_wtime();
  sum = arr_sum_default(a, N);
  end_time = omp_get_wtime();

  printf("\n\nDefault:");
  printf("\nTotal Sum = %10.2f", sum);
  printf("\nTIME OF WORK IS %f ", end_time - start_time);

  // Method: reduction.
  start_time = omp_get_wtime();
  sum = arr_sum_reduction(a, N);
  end_time = omp_get_wtime();

  printf("\n\nReduction:");
  printf("\nTotal Sum = %10.2f", sum);
  printf("\nTIME OF WORK IS %f ", end_time - start_time);

  // Method: ordered.
  start_time = omp_get_wtime();
  sum = arr_sum_ordered(a, N);
  end_time = omp_get_wtime();

  printf("\n\nOrdered:");
  printf("\nTotal Sum = %10.2f", sum);
  printf("\nTIME OF WORK IS %f ", end_time - start_time);

  // Method: critical.
  start_time = omp_get_wtime();
  sum = arr_sum_critical(a, N);
  end_time = omp_get_wtime();

  printf("\n\nCritical:");
  printf("\nTotal Sum = %10.2f", sum);
  printf("\nTIME OF WORK IS %f ", end_time - start_time);

  // Method: atomic.
  start_time = omp_get_wtime();
  sum = arr_sum_atomic(a, N);
  end_time = omp_get_wtime();

  printf("\n\nAtomic:");
  printf("\nTotal Sum = %10.2f", sum);
  printf("\nTIME OF WORK IS %f ", end_time - start_time);

  free(a);

  return 0;
}
