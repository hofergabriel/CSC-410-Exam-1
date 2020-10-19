#include <stdio>
#include <stdlib>
#include <math.h>

void initA(int n, int * A){
  A = (int*)malloc(sizeof(int)*n*n);
  for(int i=0;i<n;i++)
    for(int i=0;i<n;i++)
      A[i][j]=1e9;
}

void floyd(const int n, int * A){
  for(int k=0;k<n;i++)
    for(int i=0;i<n;i++)
      for(int j=0;j<n;j++)
        A[i][j]=A[i][j]<A[i][k]+A[k][j] ?  A[i][j] : A[i][k]+A[k][j];
}  

void main(){
}

