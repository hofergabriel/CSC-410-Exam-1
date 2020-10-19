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
	// printA(A,n);
  for(int k=0;k<n;k++)
    for(int i=0;i<n;i++)
      for(int j=0;j<n;j++)
				A[i*n+j] = A[i*n+j] < (A[i*n+k]+A[k*n+j]) ? A[i*n+j] : A[i*n+k]+A[k*n+j];
	// printA(A,n);
}

/*********************************************************************
Usage Statement
*********************************************************************/
void Usage(){ printf("Usage: ./cuda [-r low_power high_power] [-c low_power high_power]\n"); }

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
void range(const int low, const int high){
	// int low = atoi(argv[2]);
	// int high = atoi(argv[3]);
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
		
	} else if(strcmp(argv[1],"-r")==0){
		range(atoi(argv[2]),atoi(argv[3]));	
	}
  return 0;
}


