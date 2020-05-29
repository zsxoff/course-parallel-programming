#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  omp_set_num_threads(16);

  int num_threads;
  int thread_num;

  const double t1 = omp_get_wtime();

#pragma omp parallel private(num_threads, thread_num)
  {
    num_threads = omp_get_num_threads();
    thread_num = omp_get_thread_num();

    printf("OpenMP thread %d from %d threads \n", thread_num, num_threads);
  }

  double t2 = omp_get_wtime();

  // Print time.
  printf("total time: %f\n", (t2 - t1));

  return 0;
}
