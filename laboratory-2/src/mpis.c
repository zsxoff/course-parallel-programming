#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define NMAX 8

int main(int argc, char *argv[]) {
  double *a = NULL, *a_part = NULL;

  double total_sum = 0.0;
  double proc_sum = 0.0;

  double start_time;
  double end_time;

  const int N = NMAX;

  // -- get MPI routine.
  //

  MPI_Status mpi_status;

  int i;

  MPI_Init(&argc, &argv);

  int proc_num;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

  int proc_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  const int sendcount = N / proc_num;

  // -- init data.
  //

  if (proc_rank == 0) {
    a = malloc(N * sizeof(double));

    // -- manual init (hey you, change global N).

    a[0] = 2;
    a[1] = 3;
    a[2] = 1;
    a[3] = 6;
    a[4] = 3;
    a[5] = 1;
    a[6] = 0;
    a[7] = 2;

    // for (i = 0; i < N; i++)
    // {
    //   a[i] = 1.0;
    // }
  }

  // -- malloc receive parts.
  //

  a_part = malloc(sendcount * sizeof(double));

  MPI_Scatter(a,             // sendbuf
              sendcount,     // sendcount
              MPI_DOUBLE,    // sendtype
              a_part,        // recvbuf
              sendcount,     // recvcount
              MPI_DOUBLE,    // recvtype
              0,             // root
              MPI_COMM_WORLD // comm
  );

  start_time = MPI_Wtime();

  // -- get sum of parts.
  //

  for (i = 0; i < sendcount; ++i) {
    proc_sum += a_part[i];
  }

  // -- assembly.
  //

  if (proc_rank == 0) {
    total_sum = proc_sum;

    for (i = 1; i < proc_num; ++i) {
      MPI_Recv(&proc_sum,      // buf
               1,              // sendcount
               MPI_DOUBLE,     // datatype
               i,              // source
               0,              // tag
               MPI_COMM_WORLD, // comm
               &mpi_status     // status
      );

      total_sum += proc_sum;
    }
  } else {
    MPI_Send(&proc_sum,     // buf
             1,             // sendcount
             MPI_DOUBLE,    // datatype
             0,             // dest
             0,             // tag
             MPI_COMM_WORLD // comm
    );
  }

  // -- wait all procs.
  //

  MPI_Barrier(MPI_COMM_WORLD);
  end_time = MPI_Wtime();

  // -- get stats.
  //

  if (proc_rank == 0) {
    printf("\nTotal Sum = %10.2f\n", total_sum);
    printf("\nTIME OF WORK = %f\n", end_time - start_time);
    free(a);
  }

  free(a_part);

  MPI_Finalize();

  return 0;
}
