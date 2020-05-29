#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define CHUNK 100
#define NMAX 100000000
#define OMP_THREADS 16

static void sum_guided(const double *a, const double *b, double *c, const int n,
                       const int chunk) {
  int i;

#pragma omp parallel for schedule(guided, chunk) shared(a, b, c) default(none)
  for (i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

static void sum_dynamic(const double *a, const double *b, double *c,
                        const int n, const int chunk) {
  int i;

#pragma omp parallel for schedule(dynamic, chunk) shared(a, b, c) default(none)
  for (i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

static void sum_static(const double *a, const double *b, double *c, const int n,
                       const int chunk) {
  int i;

#pragma omp parallel for schedule(static, chunk) shared(a, b, c) default(none)
  for (i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

static void print_arr(const double *arr, const int n) {
  const int n_max = n < 10 ? n : 10;

  for (int i = 0; i < n_max; ++i) {
    printf("%.3f ", arr[i]);
  }
  printf("\n");
}

static void profile_methods(const double *a, const double *b, double *c,
                            const int n, const int chunk) {
  omp_set_dynamic(0);
  omp_set_num_threads(OMP_THREADS);

  printf("\nNum threads: %d, N: %d\n", OMP_THREADS, n);

  double start_time, end_time;

  // Method: guided.
  printf("\nGuided:\n");

  start_time = omp_get_wtime();
  sum_guided(a, b, c, n, chunk);
  end_time = omp_get_wtime();

  printf("\nTIME OF WORK IS: %f\n", end_time - start_time);
  print_arr(c, n);

  // Method: dynamic.
  printf("\nDynamic:\n");

  start_time = omp_get_wtime();
  sum_dynamic(a, b, c, n, chunk);
  end_time = omp_get_wtime();

  printf("\nTIME OF WORK IS: %f\n", end_time - start_time);
  print_arr(c, n);

  // Method: static.
  printf("\nStatic:\n");

  start_time = omp_get_wtime();
  sum_static(a, b, c, n, chunk);
  end_time = omp_get_wtime();

  printf("\nTIME OF WORK IS: %f\n", end_time - start_time);
  print_arr(c, n);
}

int main(int argc, char *argv[]) {
  const int n = NMAX;

  double *a = malloc(sizeof(double) * n);
  double *b = malloc(sizeof(double) * n);
  double *c = malloc(sizeof(double) * n);

  for (int i = 0; i < n; ++i) {
    a[i] = 1;
    b[i] = 2;
  }

  /*
  const int n = 5;
  double *a = malloc(sizeof(double)*n);
  double *b = malloc(sizeof(double)*n);
  double *c = malloc(sizeof(double)*n);
  a[0] = 8, b[0] = 1;
  a[1] = 4, b[1] = 2;
  a[2] = 3, b[2] = 3;
  a[3] = 2, b[3] = 4;
  a[4] = 1, b[4] = 5;
  */

  profile_methods(a, b, c, n, CHUNK);

  free(a);
  free(b);
  free(c);

  return 0;
}
