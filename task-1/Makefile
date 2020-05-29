CFLAGS=-Wall

all: mpi omp

mpi: src/mpi.c
	mpicc ${CFLAGS} $< -o $@

omp: src/omp.c
	icc ${CFLAGS} -qopenmp $< -o $@

clean:
	rm -rf ./mpi ./omp

.PHONY: all mpi omp clean
