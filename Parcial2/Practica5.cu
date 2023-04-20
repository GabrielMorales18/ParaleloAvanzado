#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <time.h>


__global__ void transposeUnroll(int* a, int* b, int n) {
    int gid = (threadIdx.x + threadIdx.y * blockDim.x) + (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y);
    int offset = blockDim.x / 2;

    for (int i = 0; i < (n * n + blockDim.x * blockDim.y - 1) / (blockDim.x * blockDim.y); i += 2)
    {
        if (gid + blockDim.x * blockDim.y * i < n * n) {
            b[(gid % n * n + gid / n) + offset * i] = a[gid + blockDim.x * blockDim.y * i];
        }
        if (gid + blockDim.x * blockDim.y * i + blockDim.x * blockDim.y < n * n) {
            b[(gid % n * n + gid / n) + offset * i + offset] = a[gid + blockDim.x * blockDim.y * i + blockDim.x * blockDim.y];
        }
    }

}

int main() {
    const int n = 8;
    int* host_a, * host_b;
    int* dev_a, * dev_b;

    host_a = (int*)malloc(n * n * sizeof(int));
    host_b = (int*)malloc(n * n * sizeof(int));

    cudaMalloc(&dev_a, n * n * sizeof(int));
    cudaMalloc(&dev_b, n * n * sizeof(int));

    srand(time(NULL));

    for (int i = 0; i < n * n; i++) {
        int r1 = (rand() % (10));
        host_a[i] = r1;
        host_b[i] = 0;
    }

    printf("Original: \n");
   
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", host_a[i * n + j]);
        }
        printf("\n");
    }

    cudaMemcpy(dev_a, host_a, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, n * n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(4, 4);

    transposeUnroll << <1, block >> > (dev_a, dev_b, n);
    cudaMemcpy(host_b, dev_b, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaDeviceReset();

    printf("Transpose: \n");

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", host_b[i * n + j]);
        }
        printf("\n");
    }

    free(host_a);
    free(host_b);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return 0;
}
