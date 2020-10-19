default:
	gcc openmp.c -fopenmp -lm -o openmp

cuda:
	nvcc cuda.cu -o cuda

openmp:
	gcc openmp.c -fopenmp -lm -o openmp

exam:
	pdflatex exam1.tex


