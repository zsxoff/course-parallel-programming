CFLAGS=-Wall

all: mpis mpir omp

mpis: src/mpis.c
	mpicc ${CFLAGS} $< -o $@

mpir: src/mpir.c
	mpicc ${CFLAGS} $< -o $@

omp: src/omp.c
	icc ${CFLAGS} -qopenmp $< -o $@

clean:
	rm -rf ./mpis ./mpir ./omp

.PHONY: all mpis mpir omp clean
