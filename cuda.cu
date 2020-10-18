#include <stdio.h>
#define THREADS_PER_BLOCK 512

__global__ void aux(int * A, const int n, const int k){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index > n*n) return;
  int i = index / n, j = index % n;
  A[i*n+j] = A[i*n+j] < (A[i*n+k]+A[k*n+j]) ? A[i*n+j] : A[i*n+k]+A[k*n+j];
}

void floyd(const int n){

  // size of A in bytes
  int Asize = n*n*sizeof(int);

  // allocate 2D array on Host
  int * A = (int *)malloc(Asize);

  // allocate 2D array on Device
  int * dA=NULL;
  cudaMalloc((void **)dA, Asize);

  // copy Array to Device
  cudaMemcpy(dA, A, Asize, cudaMemcpyHostToDevice);

  for(int k=0;k<n;k++){
    // call floyd's algorithm
    aux<<<ceil(n*n/THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(dA,n,k);
  }

  // copy Array back to Host
  cudaMemcpy(A, dA, Asize, cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(dA);
  cudaDeviceSynchronize();
}

int main() {
  int n;
  printf("size of array: ");
  scanf("size of array: %d\n", &n);
  floyd(n);
  return 0;
}

/*
  for(int k=0;k<n;i++)
    for(int i=0;i<n;i++)
      for(int j=0;j<n;j++)
        A[i][j]=A[i][j]<A[i][k]+A[k][j] ? 
          A[i][j] : A[i][k]+A[k][j];
*/




