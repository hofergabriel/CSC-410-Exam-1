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
#include <string.h>
#include <stdbool.h>
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

///////////////////////////////////////////////////////////////////////////
/*********************************************************************
Main
*********************************************************************/
void correctness(const int low, const int high){
  for(int n = pow(2,low); n <= pow(2,high); n*=2){
    int * A = makeMatrix(n);
    int * B = (int *)malloc(n*n*sizeof(int));
    int Asize = n*n*sizeof(int);
    memcpy(B, A, Asize);
    serial(B,n);
    floyd(A,n);

    bool foundDiff=false;
    for(int i=0;i<n;i++)
      for(int j=0;j<n;j++)
        if(B[i*n+j]!=A[i*n+j])
          foundDiff=true;

    free(A);
    free(B);
    if(foundDiff){
      printf("FOUND DIFFERENCE");
      return;
    }
  }
  printf("ALL SAME");
}

/*********************************************************************
Main
*********************************************************************/
void range(const int low, const int high){
  for(int n = pow(2,low); n <= pow(2,high); n*=2){
    int * A = makeMatrix(n);
    double start = omp_get_wtime();
    floyd(A,n);
	  double end = omp_get_wtime();
	  printf("%d, %f\n", n, end-start);
    free(A);
  }
}

/*********************************************************************
Main
*********************************************************************/
int main(int argc, char *argv[]) {
  if(argc==1) {
    Usage();
    return 0;
  }
  if(strcmp(argv[1],"-c")==0){
    correctness(atoi(argv[2]),atoi(argv[3])); 
  } else if(strcmp(argv[1],"-r")==0){
    range(atoi(argv[2]),atoi(argv[3])); 
  } else if(strcmp(argv[1],"-h")==0){
    Usage();
  }
  return 0;
}


