/*********************************************************************
Author: Gabriel Hofer
Date: October 19, 2020
Instructor: Dr. Karlsson
Course: CSC-410 Parallel Computing
*********************************************************************/
#include <stdio.h>
#define THREADS_PER_BLOCK 512

void printA(int * A, const int n);

/*********************************************************************
Parallel stuff, Floyd helper
*********************************************************************/
__global__ void aux(int * dA, const int n, const int k){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index > n*n) return;
	__syncthreads();
  int i = index / n;
	int j = index % n;
  dA[i*n+j] = dA[i*n+j] < (dA[i*n+k]+dA[k*n+j]) ? dA[i*n+j] : dA[i*n+k]+dA[k*n+j];
}

/*********************************************************************
Floyd-Warshall Algorithm
*********************************************************************/
void floyd(const int n){
  // size of A in bytes
  int Asize = n*n*sizeof(int);
	int inf = 512;
  // allocate 2D array on Host
  int * A = (int *)malloc(Asize);

	int tmp[n*n] = { 
		0, 2, 5, inf, inf, inf, 
		inf, 0, 7, 1, inf, 8, 
		inf, inf, 0, 4, inf, inf, 
		inf, inf, inf, 0, 3, inf, 
		inf, inf, 2, inf, 0, 3, 
		inf, 5, inf, 2, 4, 0 };

	memcpy(A,tmp,n*n*sizeof(int));

	// print before
	printA(A,n);

  // allocate 2D array on Device
  int * dA=NULL;
  cudaMalloc((void **)&dA, Asize);

  // copy Array to Device
  cudaMemcpy(dA, A, Asize, cudaMemcpyHostToDevice);

  for(int k=0;k<n;k++){
    // call floyd's algorithm
    aux<<<(n*n+THREADS_PER_BLOCK)/(THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(dA,n,k);
		cudaDeviceSynchronize();
  }

  // copy Array back to Host
  cudaMemcpy(A, dA, Asize, cudaMemcpyDeviceToHost);

	// print result
	printA(A,n);

  // Cleanup
  cudaFree(dA);
  cudaDeviceSynchronize();
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
Serial
*********************************************************************/
void serial(const int n){
	int Asize = n*n*sizeof(int);
	int inf = 512;
  // allocate 2D array on Host
  int * A = (int *)malloc(Asize);

	int tmp[n*n] = { 
		0, 2, 5, inf, inf, inf, 
		inf, 0, 7, 1, inf, 8, 
		inf, inf, 0, 4, inf, inf, 
		inf, inf, inf, 0, 3, inf, 
		inf, inf, 2, inf, 0, 3, 
		inf, 5, inf, 2, 4, 0 };

	memcpy(A,tmp,n*n*sizeof(int));

	// print before
	printA(A,n);
	for(int k=0;k<n;k++)
		for(int i=0;i<n;i++)
			for(int j=0;j<n;j++)
				A[i*n+j] = A[i*n+j] < (A[i*n+k]+A[k*n+j]) ? A[i*n+j] : A[i*n+k]+A[k*n+j];
	printA(A,n);
}

/*********************************************************************
Main
*********************************************************************/
int main() {
  int n;
	printf("size of array: ");
  cudaDeviceSynchronize();
  scanf("%d", &n);
  cudaDeviceSynchronize();
	printf("%d\n", n);
  cudaDeviceSynchronize();

  floyd(n);
	//serial(n);
  return 0;
}


