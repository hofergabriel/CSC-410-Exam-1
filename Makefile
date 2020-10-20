default:
	gcc openmp.c -fopenmp -lm -o openmp
	nvcc cuda.cu -o cuda

