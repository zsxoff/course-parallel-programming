#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  const double t1 = MPI_Wtime();

  /* Number in group */
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Group size */
  int ranksize;
  MPI_Comm_size(MPI_COMM_WORLD, &ranksize);

  printf("Hello world from process %d from total number of %d\n",
         rank,    // number in group
         ranksize // group size
  );

  const double t2 = MPI_Wtime();

  /* Print time */
  printf("proc %d sec %f\n", rank, t2 - t1);

  MPI_Finalize();

  return 0;
}
