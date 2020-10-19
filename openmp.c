/*********************************************************************
Author: Gabriel Hofer
Date: October 19, 2020
Instructor: Dr. Karlsson
Course: CSC-410 Parallel Computing
*********************************************************************/
#include <stdlib.h>
#include <math.h> 
#include <stdio.h>
#include <time.h>
#include <omp.h>
const int inf = 32768;

/*********************************************************************
Floyd's Algorithm, OpenMP
*********************************************************************/
void floyd(int * A, const int n){
  for(int k=0;k<n;k++)
		#pragma omp parallel for
    for(int i=0;i<n;i++)
			#pragma omp parallel for
      for(int j=0;j<n;j++)
				A[i*n+j] = A[i*n+j] < (A[i*n+k]+A[k*n+j]) ? A[i*n+j] : A[i*n+k]+A[k*n+j];
}

/*********************************************************************
Serial, used for checking correctness 
*********************************************************************/
void serial(int * A, const int n){
	// printA(A,n);
  for(int k=0;k<n;k++)
    for(int i=0;i<n;i++)
      for(int j=0;j<n;j++)
				A[i*n+j] = A[i*n+j] < (A[i*n+k]+A[k*n+j]) ? A[i*n+j] : A[i*n+k]+A[k*n+j];
	// printA(A,n);
}

/*********************************************************************
Make Random Matrix 
*********************************************************************/
int * makeMatrix(const int n){
	int * A = (int *)malloc(n*n*sizeof(int));
	srand(time(0));
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++){
			if(rand()&1) A[i*n+j]=(rand()%20)+1; // random number in range [1,20]
			else A[i*n+j]=inf;
		}
	return A;
}

/*********************************************************************
Usage Statement
*********************************************************************/
void Usage(){ printf("Usage: ./cuda -N n_integer\n"); }

/*********************************************************************
Main
*********************************************************************/
int main(int argc, char *argv[]) {
	if(argc==1) {
		Usage();
		return 0;
	}
	// convert n from cstring to integer
  int n = atoi(argv[2]);
	// allocate memory for graph
	int * A = makeMatrix(n);

	double start = omp_get_wtime();
  serial(A,n);
	double end = omp_get_wtime();
	printf("Serial Execution Time:   %f\n", end-start);

  start = omp_get_wtime();
  floyd(A,n);
	end = omp_get_wtime();
	printf("Parallel Execution Time: %f\n", end-start);

	free(A);	
  return 0;
}






