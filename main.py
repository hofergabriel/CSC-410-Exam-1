"""
Author: Gabriel Hofer
Instructor: Dr. Karlsson
Due: October 19, 2020

"""
import numpy as np


def floyd(n):
  A=np.zeros((n,n)) 
  for k in range(n):
    for i in range(n):
      for j in range(n):
        A[i,j]=min(A[i,j],A[i,k]+A[k,j])

def main(): pass

