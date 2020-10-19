/*********************************************************************
Author: Gabriel Hofer
Date: October 19, 2020
Instructor: Dr. Karlsson
Course: CSC-410 Parallel Computing
*********************************************************************/
#include <stdio.h>
#include <time.h>
#define THREADS_PER_BLOCK 512
const int inf = 32768;

void printA(int * A, const int n);

/*********************************************************************
Floyd helper (auxiliary)
*********************************************************************/
__global__ void aux(int * dA, const int n, const int k){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index > n*n) return;
  __syncthreads();
  int i = index / n, j = index % n;
  dA[i*n+j] = dA[i*n+j] < (dA[i*n+k]+dA[k*n+j]) ? dA[i*n+j] : dA[i*n+k]+dA[k*n+j];
}

/*********************************************************************
Floyd-Warshall Algorithm
*********************************************************************/
void floyd(int * dA, const int n){
  for(int k=0;k<n;k++){
    aux<<<(n*n+THREADS_PER_BLOCK)/(THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(dA,n,k);
    cudaDeviceSynchronize();
  }
}

/*********************************************************************
Print Array
*********************************************************************/
void printA(int * A, const int n){
  for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
			printf("%d\t",A[i*n+j]);
			cudaDeviceSynchronize();
		}
		printf("\n");
		cudaDeviceSynchronize();
	}
	printf("\n");
	cudaDeviceSynchronize();
}

/*********************************************************************
Serial, used for checking correctness 
*********************************************************************/
void serial(int * A, const int n){
  for(int k=0;k<n;k++)
    for(int i=0;i<n;i++)
      for(int j=0;j<n;j++)
				A[i*n+j] = A[i*n+j] < (A[i*n+k]+A[k*n+j]) ? A[i*n+j] : A[i*n+k]+A[k*n+j];
}

/*********************************************************************
Usage Statement
*********************************************************************/
void Usage(){ 
	printf("Usage: ./cuda [-r low_power high_power] [-c low_power high_power]\n"); 
	printf("\t-r: runs Floyd's algorithm in parallel on range of powers of two\n");
	printf("\t-c: runs correctness tests on range of powers of two\n");
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
Main
*********************************************************************/
void correctness(const int low, const int high){
	for(int n = pow(2,low); n <= pow(2,high); n*=2){
		int * A = makeMatrix(n);
		int * B = (int *)malloc(n*n*sizeof(int));
  	int Asize = n*n*sizeof(int);
		memcpy(B, A, Asize);
		serial(B,n);
		
  	int * dA=NULL;
  	cudaMalloc((void **)&dA, Asize);
  	cudaMemcpy(dA, A, Asize, cudaMemcpyHostToDevice);
  	floyd(dA,n);
  	cudaMemcpy(A, dA, Asize, cudaMemcpyDeviceToHost);

		bool foundDiff=false;
		for(int i=0;i<n;i++)
			for(int j=0;j<n;j++)
				if(B[i*n+j]!=A[i*n+j])
					foundDiff=true;

  	cudaFree(dA);
		free(A);
		free(B);
  	cudaDeviceSynchronize();
		if(foundDiff){
			printf("FOUND DIFFERENCE");
			break;
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
  	int Asize = n*n*sizeof(int);
  	int * dA=NULL;
  	cudaMalloc((void **)&dA, Asize);
  	cudaMemcpy(dA, A, Asize, cudaMemcpyHostToDevice);
		clock_t before = clock();
  	floyd(dA,n);
		clock_t after = clock();
		printf("Execution Time: %f\n", (float)(after-before)/CLOCKS_PER_SEC);
  	cudaMemcpy(A, dA, Asize, cudaMemcpyDeviceToHost);
  	cudaFree(dA);
		free(A);
  	cudaDeviceSynchronize();
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


