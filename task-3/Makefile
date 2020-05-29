CFLAGS=-Wall

all: mpis mpiv omp

mpis: src/mpis.c
	mpicc ${CFLAGS} $< -o $@

mpiv: src/mpiv.c
	mpicc ${CFLAGS} $< -o $@

omp: src/omp.c
	icc ${CFLAGS} -qopenmp $< -o $@

clean:
	rm -rf ./mpis ./mpiv ./omp

.PHONY: all mpis mpiv omp clean
