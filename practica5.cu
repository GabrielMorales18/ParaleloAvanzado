//Practica 1 - convolución 2D

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n\r", cudaGetErrorString(code), file, line);
        if (abort)exit(code);
    }
}

__global__ void convolucion(int* a, int* k, int* b, int n, int m, int size) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    int suma = 0;
    if (row > 0 && row < m - 1 && col>0 && col < n - 1) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                suma += (a[(row - 1) * m + i + (col - 1) + j] * k[i * size + j]); 
            }
        }
        b[row * m + col] = suma;
    }
}

int main() {

    const int n = 32, m = 32, kernel = 3;
    int* host_a, * host_b, * host_k;
    int* dev_a, * dev_b, * dev_k;

    host_a = (int*)malloc(n * m * sizeof(int));
    host_b = (int*)malloc(n * m * sizeof(int));
    host_k = (int*)malloc(kernel * kernel * sizeof(int));

    cudaMalloc(&dev_a, n * m * sizeof(int));
    cudaMalloc(&dev_b, n * m * sizeof(int));
    cudaMalloc(&dev_k, kernel * kernel * sizeof(int));

    srand(time(0));

    for (int i = 0; i < n * m; i++) {
        int r1 = (rand() % (3));
        host_a[i] = r1;
        host_b[i] = r1;
    }

    host_k[0] = 0;
    host_k[1] = 0;
    host_k[2] = 0;
    host_k[3] = 0;
    host_k[4] = 1;
    host_k[5] = 0;
    host_k[6] = 0;
    host_k[7] = 0;
    host_k[8] = 0;

    cudaMemcpy(dev_a, host_a, n * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, n * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k, host_k, kernel * kernel * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(8, 8);

    convolucion << <1, block >> > (dev_a, dev_k, dev_b, n, m, kernel);
    cudaMemcpy(host_b, dev_b, n * m * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaDeviceReset();

    cout << "A:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cout << host_a[i * m + j] << " ";
        }
        cout << "\n";
    }

    cout << "Res:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cout << host_b[i * n + j] << " ";
        }
        cout << "\n";
    }

    free(host_a);
    free(host_b);
    free(host_k);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_k);

    return 0;
}
