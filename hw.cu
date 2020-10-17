#include <stdio.h>

__global__ void print_kernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void floyd(const int n, int * A){
  for(int k=0;k<n;i++)
    for(int i=0;i<n;i++)
      for(int j=0;j<n;j++)
        A[i][j]=A[i][j]<A[i][k]+A[k][j] ? 
          A[i][j] : A[i][k]+A[k][j];
}

int main() {
    print_kernel<<<1,1>>>();
    cudaDeviceSynchronize();
}
