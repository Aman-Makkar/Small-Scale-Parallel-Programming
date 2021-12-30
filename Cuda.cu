#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <iostream>
#include <cuda_runtime.h>  // For CUDA runtime API
#include "helper_cuda.h"  // For checkCudaError macro
#include "helper_timer.h"  // For CUDA SDK timers

#define BLOCK_SIZE 16
const dim3 BLOCK_DIM(BLOCK_SIZE, BLOCK_SIZE);

void MatrixMultiply(int matrix_size, float *A, float *B, float *C)
   {
       int i,j,k;
       // printf("Resultant Matrix:\n");
       for (i = 0; i < matrix_size; i++) {
           for (j = 0; j < matrix_size; j++) {
               float t = 0.0f;
               for (k = 0; k < matrix_size; k++) {
                   int idxA = i * matrix_size + k;
                   int idxB = k * matrix_size + j;
                   t += A[idxA] * B[idxB];
               }
               C[i] = t;
               // printf("%f   ", C[i]);
           }
           // printf("\n");
       }
   }

void transposed(int matrix_size, float *transpose, float *B)
{
       int j, k;
       for (j = 0; j < matrix_size; j++) {
           for (k = 0; k < matrix_size; k++) {
               transpose[k * matrix_size + j] = B[j * matrix_size + k];
           }
       }
   }

void print_matrix(int matrix_size, float *A, float *B, float *C)
   {
       int i, j;
       printf("The resultant Matrix is:\n");
       for(i = 0; i < matrix_size; i++)
       {
           for(j = 0; j < matrix_size; j++)
           {
              int idx = i * matrix_size + j;
              printf("%f   ", C[idx]);
           }
           printf("\n");
       }
   }

__global__ void gpu_matrix_mult(float *A,float *B, float *C, int matrix_size)
{
       int row = blockIdx.y * blockDim.y + threadIdx.y;
       int col = blockIdx.x * blockDim.x + threadIdx.x;
       float sum = 0.0f;
       if( col < matrix_size && row < matrix_size)
       {
           for(int i = 0; i < matrix_size; i++)
           {
               sum += A[row * matrix_size + i] * B[i * matrix_size + col];
           }
           C[row * matrix_size + col] = sum;
       }
   }

__global__ void square_matrix_mult_gpu (float *A, float *B, float *C, int matrix_size)
   {
     __shared__ float ax[BLOCK_SIZE][BLOCK_SIZE];
     __shared__ float ay[BLOCK_SIZE][BLOCK_SIZE];

     int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
     int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
     float t = 0.0f;
     int idx;

     for(int i =0; i<gridDim.x; i++){
       idx = row * matrix_size + i * BLOCK_SIZE + threadIdx.x;

       ax[threadIdx.y][threadIdx.x] = A[idx];
       ay[threadIdx.y][threadIdx.x] = B[idx];

        __syncthreads();

       for(int k = 0; k <BLOCK_SIZE; k++)
       {
         t += ax[threadIdx.y][k] * ay[k][threadIdx.x];
       }
          __syncthreads();
     }
     if(row < matrix_size && col < matrix_size){
       C[row * matrix_size + col] = t;
     }
   }

int main(int argc, char** argv) {

    int matrix_size;
    if(argc < 2)
        fprintf(stderr, "Usage: %s  matrix_size, it is a square matrix\n",argv[0]);
    matrix_size = atoi(argv[1]);

    float* A = new float[matrix_size * matrix_size];
    float* B = new float[matrix_size * matrix_size];
    float* C = new float[matrix_size * matrix_size];
    float* transposeA = new float[matrix_size * matrix_size];
    float* transposeB = new float[matrix_size * matrix_size];
    float* SerialC = new float[matrix_size * matrix_size];

    srand(time(0));
    // printf("Matrix A:\n");
    for (int row = 0; row < matrix_size; ++row) {
      for (int col = 0; col < matrix_size; ++col) {
        int idx = row * matrix_size + col;
        A[idx] = 100.0f * static_cast<float>(rand()) / RAND_MAX;
        // printf("%2.2f   ", A[idx]);
      }
      // printf("\n");
    C[row] = 0.0f;
}
    // printf("Matrix B:\n");
    for (int row = 0; row < matrix_size; ++row) {
    for (int col = 0; col < matrix_size; ++col) {
      int idx = row * matrix_size + col;
      B[idx] = 100.0f * static_cast<float>(rand()) / RAND_MAX;
      // printf("%2.2f   ", B[idx]);
  }
  // printf("\n");
}

    transposed(matrix_size, transposeB, B);
    transposed(matrix_size, transposeA, A);
    float *d_a, *d_x, *d_y;

     checkCudaErrors(cudaMalloc((void**) &d_a, matrix_size * matrix_size * sizeof(float)));
     checkCudaErrors(cudaMalloc((void**) &d_x, matrix_size * matrix_size * sizeof(float)));
     checkCudaErrors(cudaMalloc((void**) &d_y, matrix_size * matrix_size * sizeof(float)));

     checkCudaErrors(cudaMemcpy(d_a, transposeA, matrix_size * matrix_size * sizeof(float), cudaMemcpyHostToDevice));
     checkCudaErrors(cudaMemcpy(d_x, transposeB,  matrix_size * matrix_size * sizeof(float), cudaMemcpyHostToDevice));

     // ------------------------ Calculations on the CPU ------------------------- //
     float flopcnt=2.e-6*matrix_size*matrix_size;

     // Create the CUDA SDK timer.
     StopWatchInterface* timer = 0;
     sdkCreateTimer(&timer);

     timer->start();
     MatrixMultiply(matrix_size, transposeA, transposeB, SerialC);
     // print_matrix(matrix_size, A, B, SerialC);

     timer->stop();
     float cpuflops=flopcnt/ timer->getTime();
     printf("CPU time = %f ms, GFLOPS = %f\n", timer->getTime(), cpuflops);

     // ------------------------ Calculations on the GPU ------------------------- //

  // Calculate the dimension of the grid of blocks. A 1D grid suffices.
  // const dim3 GRID_DIM(matrix_size/BLOCK_SIZE, matrix_size/BLOCK_SIZE);
  unsigned int grid_matrix = (matrix_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const dim3 GRID_DIM(grid_matrix, grid_matrix);

  timer->reset();
  timer->start();
  square_matrix_mult_gpu<<<GRID_DIM, BLOCK_DIM >>>(d_a, d_x, d_y, matrix_size);
  checkCudaErrors(cudaDeviceSynchronize());

  timer->stop();
  float gpuflops=flopcnt/ timer->getTime();
  printf("GPU time = %f ms, GFLOPS = %f\n", timer->getTime(), gpuflops);

  // TODO Download the resulting vector d_y from the device and store it in h_y_d.
  checkCudaErrors(cudaMemcpy(C, d_y, (matrix_size * matrix_size) * sizeof(float),cudaMemcpyDeviceToHost));
  // for(int i = 0; i < matrix_size; i++){
  //   for(int j = 0; j < matrix_size; j++) {
  //     printf("%f   ", C[i * matrix_size + j]);
  //   }
  //   printf("\n");
  // }

  // Now let's check if the results are the same.
  float reldiff = 0.0f;
  float diff = 0.0f;

  for (int row = 0; row < matrix_size; row++) {
    float maxabs = max(abs(SerialC[row]), abs(C[row]));
    if (maxabs == 0.0f) maxabs = 1.0f;
    reldiff = max(reldiff, abs(SerialC[row] - C[row])/maxabs);
    diff = max(diff, abs(SerialC[row] - C[row]));
  }
  printf("Max diff = %2.2f, Max rel diff = %2.2f\n", diff, reldiff);

// ------------------------------- Cleaning up ------------------------------ //

  delete timer;

  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] SerialC;
  delete[] transposeA;
  delete[] transposeB;
  return 0;
}


