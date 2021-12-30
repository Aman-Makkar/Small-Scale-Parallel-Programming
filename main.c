#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdbool.h>

#define CHUNK 5

void MatrixMultiply(int matrix_size, double A[matrix_size][matrix_size], double transposeB[matrix_size][matrix_size], double C[matrix_size][matrix_size]);
void transposed(int matrix_size, double transpose[matrix_size][matrix_size], double B[matrix_size][matrix_size]);
void print_matrix(int matrix_size, double A[matrix_size][matrix_size], double B[matrix_size][matrix_size], double C[matrix_size][matrix_size]);
void MatrixMultiplyOmp(int matrix_size, double A[matrix_size][matrix_size], double transposeB[matrix_size][matrix_size], double C[matrix_size][matrix_size]);
void checkResults(int matrix_size, double C[matrix_size][matrix_size], double SerialC[matrix_size][matrix_size]);

int main(int argc, char** argv)
{
    int i, j, matrix_size;
    //To ensure that the right amount of arguments are provided when running the program
    if(argc < 2)
        fprintf(stderr, "Usage: %s  row/col, it is a square matrix\n",argv[0]);
    matrix_size = atoi(argv[1]);

    int tid, nthreads;

    //Dynamic memory allocation for the matrices
    double (*A)[matrix_size] = malloc(sizeof(double)*matrix_size*matrix_size);
    double (*B)[matrix_size] = malloc(sizeof(double)*matrix_size*matrix_size);
    double (*C)[matrix_size] = malloc(sizeof(double)*matrix_size*matrix_size);
    double (*transposeB)[matrix_size] = malloc(sizeof(double)*matrix_size*matrix_size);
    double (*SerialC)[matrix_size] = malloc(sizeof(double)*matrix_size*matrix_size);

    //Randomising the A and B matrices
    srand(time(0));
    for(i = 0; i < matrix_size; i++){
        for(j = 0; j < matrix_size; j++) {
            A[i][j] = 100.0f * ((double) rand()) / RAND_MAX;;
            B[i][j] = 100.0f * ((double) rand()) / RAND_MAX;
        }
    }

// Only left here if the user wants to print the randomised matrices.
//    printf("Matrix A: \n");
//    for(i = 0; i < matrix_size; i++) {
//        for (j = 0; j < matrix_size; j++) {
//            printf("%2.2f ", A[i][j]);
//        }
//        printf("\n");
//    }
//    printf("Matrix B: \n");
//    for(i = 0; i < matrix_size; i++) {
//        for (j = 0; j < matrix_size; j++) {
//            printf("%2.2f ", B[i][j]);
//        }
//        printf("\n");
//    }

    //Transposing matrix B
    transposed(matrix_size, transposeB, B);
    //To start measuring the time of the parallel execution
    double startTime = omp_get_wtime();
    //To carry out the multiplication of A * B and save it into matrix C using OpenMP.
    MatrixMultiplyOmp(matrix_size, A, transposeB, C);
    //To end measuring the time of the parallel execution
    double endTime = omp_get_wtime();
    //Only left here if the user wants to print the resultant matrix C.
//    print_matrix(matrix_size, A, transposeB, C);
    //To start measuring the time of the serial execution
    double startTimeSer = omp_get_wtime();
    //To carry out the multiplication of A * B and save it into matrix C using serial algorithm.
    MatrixMultiply(matrix_size, A, transposeB, SerialC);
    //To end measuring the time of the serial execution
    double endTimeSer = omp_get_wtime();
    //Only left here if the user wants to print the resultant matrix C.
//    print_matrix(matrix_size, A, transposeB, SerialC);

    // To measure the total execution time of the parallel algorithm.
    double totTime = endTime - startTime;
    // To measure the total execution time of the serial algorithm.
    double totTimeser = endTimeSer - startTimeSer;

    //To compare the results of the parallel and serial algorithm, ensuring they match.
    checkResults(matrix_size, C, SerialC);
    printf("Parallel time = %f and Serial time = %f\n", totTime, totTimeser);

    //To measure the performance of the two algorithms, in Mega flops.
    double mflops = (2.0e-6)*matrix_size*matrix_size*matrix_size/totTime;
#pragma omp parallel
  {
#pragma omp master
    {
      fprintf(stdout,"Multiplying matrices of size %d x %d (%d) (block_42) with %d threads: time %lf  MFLOPS %lf \n",
	      matrix_size,matrix_size,matrix_size,omp_get_num_threads(),totTime,mflops);
    }
  }
  // To free the allocated memory.
    free(A);
    free(B);
    free(C);
    free(transposeB);
    free(SerialC);
    return 0;

}
//Serial algorithm to carry out the multiplication of matrix A and B and saves it matrix C.
void MatrixMultiply(int matrix_size, double A[matrix_size][matrix_size], double transposeB[matrix_size][matrix_size], double C[matrix_size][matrix_size])
{
    int i,j,k;
    for (i = 0; i < matrix_size; i++) {
        for (j = 0; j < matrix_size; j++) {
            C[i][j] = 0;
            for (k = 0; k < matrix_size; k++) {
                C[i][j] = C[i][j] + A[i][k] * transposeB[j][k];
            }
        }
    }
}

//Transposes a matrix.
void transposed(int matrix_size, double transpose[matrix_size][matrix_size], double B[matrix_size][matrix_size])
{
    int j, k;
    for (j = 0; j < matrix_size; j++) {
        for (k = 0; k < matrix_size; k++) {
            transpose[k][j] = B[j][k];
        }
    }
}

//To print a matrix.
void print_matrix(int matrix_size, double A[matrix_size][matrix_size], double B[matrix_size][matrix_size], double C[matrix_size][matrix_size])
{
    int i, j, k;
    for(i = 0; i < matrix_size; i++)
    {
        for(j = 0; j < matrix_size; j++)
        {
            printf("%2.2f   ", C[i][j]);
        }
        printf("\n");
    }
}
//Parallel algorithm to carry out the multiplication of matrix A and B and saves it matrix C, uses OpenMP.
void MatrixMultiplyOmp(int matrix_size, double A[matrix_size][matrix_size], double transposeB[matrix_size][matrix_size], double C[matrix_size][matrix_size])
{
    int i,j,k;
    double dot = 0.0;

#pragma omp parallel for shared(A, transposeB, C) private(i, j, k) schedule(dynamic, CHUNK) reduction(+: dot)
    for (i = 0; i < matrix_size; i++) {
        for (j = 0; j < matrix_size; j++) {
            C[i][j] = 0;
            for (k = 0; k < matrix_size; k++) {
                dot += (A[i][k] * transposeB[j][k]);
                C[i][j] = dot;
            }
        }
    }
}

//To check the results of two resultant matrices match.
void checkResults(int matrix_size, double C[matrix_size][matrix_size], double SerialC[matrix_size][matrix_size])
{
    int i, j;
    bool answer = true;
    for(i = 0; i < matrix_size; i++)
    {
        for(j = 0; j < matrix_size; j++)
        {
            if ((SerialC[i][j] - C[i][j]) > 0.1)
                answer = false;
        }
    }
    if (answer) {
        printf("The values for Serial and Parallel matrices match!\n");
    } else
        printf("The values for Serial and Parallel matrices do not match!\n");
}