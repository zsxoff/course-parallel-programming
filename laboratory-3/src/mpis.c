#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define NMAX 8

static void print_arr(const double *arr, const int n) {
  const int n_max = n < 10 ? n : 10;

  for (int i = 0; i < n_max; ++i) {
    printf("%.3f ", arr[i]);
  }
  printf("\n");
}

int main(int argc, char *argv[]) {
  double *a = NULL, *a_part = NULL;
  double *b = NULL, *b_part = NULL;
  double *c = NULL, *c_part = NULL;

  const int N = NMAX;

  double start_time;
  double end_time;

  // Get MPI routine.
  int i;

  MPI_Init(&argc, &argv);

  int proc_num;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

  int proc_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  if (proc_rank == 0) {
    printf("\nNMAX: %d, proc_num: %d\n", NMAX, proc_num);
  }

  const int sendcount = N / proc_num;

  // Init data.
  if (proc_rank == 0) {
    a = malloc(sizeof(double) * N);
    b = malloc(sizeof(double) * N);
    c = malloc(sizeof(double) * N);

    /*
    // Manual init.
    // ! WARNING ! Change global NMAX before init.
    a[0] = 1, b[0] = 2;
    a[1] = 2, b[1] = 4;
    a[2] = 1, b[2] = 1;
    a[3] = 3, b[3] = 4;
    a[4] = 2, b[4] = 2;
    a[5] = 3, b[5] = 3;
    a[6] = 1, b[6] = 2;
    a[7] = 4, b[7] = 1;
    */

    for (i = 0; i < N; ++i) {
      a[i] = 1.0;
      b[i] = 2.0;
    }
  }

  // Malloc receive parts.
  a_part = (double *)malloc(sendcount * sizeof(double));
  b_part = (double *)malloc(sendcount * sizeof(double));
  c_part = (double *)malloc(sendcount * sizeof(double));

  start_time = MPI_Wtime();

  // Sends on procs.
  MPI_Scatter(a,             // *sendbuf
              sendcount,     // sendcount
              MPI_DOUBLE,    // sendtype
              a_part,        // *recvbuf
              sendcount,     // recvcount
              MPI_DOUBLE,    // recvtype
              0,             // root
              MPI_COMM_WORLD // comm
  );

  MPI_Scatter(b,             // *sendbuf
              sendcount,     // sendcount
              MPI_DOUBLE,    // sendtype
              b_part,        // *recvbuf
              sendcount,     // recvcount
              MPI_DOUBLE,    // recvtype
              0,             // root
              MPI_COMM_WORLD // comm
  );

  // Get sum of parts.
  for (i = 0; i < sendcount; i++) {
    c_part[i] = a_part[i] + b_part[i];
  }

  // Assembly.
  MPI_Gather(c_part,        // *sendbuf
             sendcount,     // sendcount
             MPI_DOUBLE,    // sendtype
             c,             // *recvbuf
             sendcount,     // recvcount
             MPI_DOUBLE,    // recvtype
             0,             // root
             MPI_COMM_WORLD // comm
  );

  MPI_Barrier(MPI_COMM_WORLD);
  end_time = MPI_Wtime();

  if (proc_rank == 0) {
    printf("\nA: ");
    print_arr(a, N);

    printf("\nB: ");
    print_arr(b, N);

    printf("\nN: ");
    print_arr(c, N);

    printf("\nTIME OF WORK IS %f\n", end_time - start_time);

    free(a);
    free(b);
    free(c);
  }

  free(a_part);
  free(b_part);
  free(c_part);

  MPI_Finalize();

  return 0;
}
