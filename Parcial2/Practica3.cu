#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <time.h>

__global__ void search(int* a, int n, int* pos, int find) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        if (a[tid] == find) {
            *pos = tid;
        }
    }
}
int main() {
    int size = 32;
    int find;
    int* host_a, * res, * pos;
    int* dev_a, * dev_pos;

    host_a = (int*)malloc(size * sizeof(int));
    pos = (int*)malloc(sizeof(int));
    pos[0] = -1;
    res = (int*)malloc(size * sizeof(int));

    cudaMalloc(&dev_a, size * sizeof(int));
    cudaMalloc(&dev_pos, sizeof(int));

    srand(time(NULL));

    for (int i = 0; i < size; i++) {
        int r1 = (rand() % (256));
        host_a[i] = r1;
        printf("%d ", host_a[i]);
    }

    printf("\n");

    cudaMemcpy(dev_a, host_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pos, pos, sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(size >= 1024 ? size / 1024 : 1);
    dim3 block(1024);

    printf("Valor a encontrar: ");
    scanf("%d", &find);

    search << <grid, block >> > (dev_a, size, dev_pos, find);

    cudaDeviceSynchronize();

    cudaMemcpy(pos, dev_pos, sizeof(int), cudaMemcpyDeviceToHost);

    if (pos[0] == -1) {
        printf("No se encontro el valor\n");
    }
    else {
        printf("Encontrado en el indice %d \n", pos[0]);
    }

    free(host_a);
    free(pos);
    free(res);
    cudaFree(dev_a);
    cudaFree(dev_pos);

    return 0;
}
