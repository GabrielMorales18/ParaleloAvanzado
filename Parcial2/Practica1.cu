#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;

__global__ void transpuesta(int* a, int* b, int n) {

    __shared__ int s[64];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int offsetBlock = blockIdx.x * blockDim.x * blockDim.y;
    int offsetRow = blockIdx.y * blockDim.x * blockDim.y * gridDim.x;
    int gid = tid + offsetBlock + offsetRow;
    int row = gid / n;
    int col = gid - row * n;

    if (gid < n * n) {
        s[row * n + col] = a[row * n + col];
        __syncthreads();
        b[col * n + row] = s[row * n + col];
    }
}

__global__ void convolucion2D(int* a, int* b, int* k, int n, int m, int kernelSize) {

    __shared__ int s[64];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int offsetBlock = blockIdx.x * blockDim.x * blockDim.y;
    int offsetRow = blockIdx.y * blockDim.x * blockDim.y * gridDim.x;
    int gid = tid + offsetBlock + offsetRow;
    int row = gid / n;
    int col = gid - row * n;
    int suma = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (row + i >= 0 && row + i < n && col + j >= 0 && col + j < n) {
                s[(i + 1) * kernelSize + j + 1] = k[(i + 1) * kernelSize + j + 1];
                __syncthreads();
                suma += a[(row + i) * m + col + j] * s[(i + 1) * kernelSize + j + 1];

            }
        }
    }

    b[row * m + col] = suma;

}

int main() {

    const int sIzeK = 3, n = 8, m = 8;
    int* hostA_K, * hostB_K, * host_a, * host_b;
    int* devA_K, * devB_K, * dev_a, * dev_b;

    hostA_K = (int*)malloc(sIzeK * sIzeK * sizeof(int));
    hostB_K = (int*)malloc(sIzeK * sIzeK * sizeof(int));
    host_a = (int*)malloc(n * m * sizeof(int));
    host_b = (int*)malloc(n * m * sizeof(int));

    cudaMalloc(&devA_K, sIzeK * sIzeK * sizeof(int));
    cudaMalloc(&devB_K, sIzeK * sIzeK * sizeof(int));
    cudaMalloc(&dev_a, n * m * sizeof(int));
    cudaMalloc(&dev_b, n * m * sizeof(int));

    srand(time(NULL));

    for (int i = 0; i < sIzeK * sIzeK; i++) {
        int r1 = (rand() % (1));
        hostA_K[i] = r1;
        hostB_K[i] = 0;
    }

    for (int i = 0; i < n * m; i++) {
        int r1 = (rand() % (3));
        host_a[i] = r1;
        host_b[i] = 0;
    }

    
    hostA_K[3] = 1;

    printf("Matrix Orig: \n");

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d ", host_a[i * m + j]);
        }
        printf("\n");
    }

    printf("\n");

    cudaMemcpy(devA_K, hostA_K, sIzeK * sIzeK * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devB_K, hostB_K, sIzeK * sIzeK * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_a, host_a, n * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, n * m * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid(32 / (sIzeK * sIzeK), 32 / (sIzeK * sIzeK));

    transpuesta << <grid, block >> > (devA_K, devB_K, sIzeK);
    cudaMemcpy(hostB_K, devB_K, sIzeK * sIzeK * sizeof(int), cudaMemcpyDeviceToHost);;

    dim3 block2(32, 32);
    dim3 grid2((64 + (n * m) - 1) / (n * m), (64 + (n * m) - 1) / (n * m));

    convolucion2D << <grid2, block2 >> > (dev_a, dev_b, devB_K, n, m, sIzeK);
    cudaMemcpy(host_b, dev_b, n * m * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaDeviceReset();


    printf("Matrix Res: \n");

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d ", host_b[i * m + j]);
        }
        printf("\n");
    }

    free(hostA_K);
    free(hostB_K);
    cudaFree(devA_K);
    cudaFree(devB_K);

    return 0;
}
