//practica 1
/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void idx_calc_gid()
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    //gid = tid + offset
    int offsetBlock = blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
    int offsetRow = blockIdx.y * gridDim.x * blockDim.x * blockDim.y * blockDim.z;
    int offsetDepth = blockIdx.z * blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y;
    int gid = tid + offsetBlock + offsetRow + offsetDepth;
    printf("[DEVICE]  gid: %d\n\r", gid);
}

int main()
{;

    dim3 grid(2, 2, 2);
    dim3 block(2, 2, 2);

    idx_calc_gid << <grid, block >> > ();

    cudaDeviceSynchronize();

    cudaDeviceReset();


    return 0;
}
*/

//practica 2
/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


__global__ void sum_array_gpu(int* a, int* b, int* c, int size) {
    int totalThreads = blockDim.x * blockDim.y * blockDim.z;
    int tid = threadIdx.x //1D
        + threadIdx.y * blockDim.x //2D
        + threadIdx.z * blockDim.x * blockDim.y; //3D

    int bid = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + blockIdx.z * gridDim.x * gridDim.y; //3D

    int gid = tid + bid * totalThreads; // threadIdx + offset

    if (gid < size) {
        c[gid] = a[gid] + b[gid];
    }

}

void sum_array_cpu(int* a, int* b, int* c, int size) {
    
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }

}

int main()
{
    
    const int size = 10000;
    int* host_a, *host_b, *host_c, *check; 
    int* dev_a, *dev_b, *dev_c;

    host_a = (int*)malloc(size * sizeof(int));
    host_b = (int*)malloc(size * sizeof(int));
    host_c = (int*)malloc(size * sizeof(int));
    check = (int*)malloc(size * sizeof(int));

    cudaMalloc(&dev_a, size * sizeof(int));
    cudaMalloc(&dev_b, size * sizeof(int));
    cudaMalloc(&dev_c, size * sizeof(int));

    for (int i = 0; i < size; i++) {
        int r1 = (rand() % (256));
        int r2 = (rand() & (256));

        host_a[i] = r1;
        host_b[i] = r2;
        host_c[i] = 0;
    }

    cudaMemcpy(dev_a, host_a, size*sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(dev_b, host_b, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, host_c, size * sizeof(int), cudaMemcpyHostToDevice);

    sum_array_cpu(host_a, host_b, host_c, size);
    sum_array_gpu << <79, 128 >> > (dev_a, dev_b, dev_c, size); 
    cudaMemcpy(check, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaDeviceReset();

    bool equal = true;
    for (int i = 0; i < size; i++) {
        if (host_c[i] != check[i]) {
            equal = false;
        }
    }
    
    if (equal) {
        printf("Arrays are equal\n");
    }
    else {
        printf("Arrays are different\n");
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(host_a);
    cudaFree(host_b);
    cudaFree(host_c);
    cudaFree(check);

    return 0;
}
*/

//practica 3

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


__global__ void sum_array_gpu(int* a, int* b, int* c, int* d,int size) {
    int totalThreads = blockDim.x * blockDim.y * blockDim.z;
    int tid = threadIdx.x //1D
        + threadIdx.y * blockDim.x //2D
        + threadIdx.z * blockDim.x * blockDim.y; //3D

    int bid = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + blockIdx.z * gridDim.x * gridDim.y; //3D

    int gid = tid + bid * totalThreads; // threadIdx + offset

    if (gid < size) {
        d[gid] = a[gid] + b[gid] + c[gid];
    }

}

void sum_array_cpu(int* a, int* b, int* c, int* d, int size) {
    
    for (int i = 0; i < size; i++) {
        d[i] = a[i] + b[i] + c[i];
    }

}

int main()
{
    
    const int size = 10000;
    int* host_a, *host_b, *host_c, *host_d, *check; 
    int* dev_a, *dev_b, *dev_c, *dev_d;

    host_a = (int*)malloc(size * sizeof(int));
    host_b = (int*)malloc(size * sizeof(int));
    host_c = (int*)malloc(size * sizeof(int));
    host_d = (int*)malloc(size * sizeof(int));
    check = (int*)malloc(size * sizeof(int));

    cudaMalloc(&dev_a, size * sizeof(int));
    cudaMalloc(&dev_b, size * sizeof(int));
    cudaMalloc(&dev_c, size * sizeof(int));
    cudaMalloc(&dev_d, size * sizeof(int));

    for (int i = 0; i < size; i++) {
        int r1 = (rand() % (256));
        int r2 = (rand() & (256));
        int r3 = (rand() & (256));

        host_a[i] = r1;
        host_b[i] = r2;
        host_c[i] = r3;
        host_d[i] = 0;
    }

    cudaMemcpy(dev_a, host_a, size*sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(dev_b, host_b, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, host_c, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_d, host_d, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(4, 4, 8);
    dim3 block(4, 4, 5);


    sum_array_cpu(host_a, host_b, host_c, host_d, size);
    sum_array_gpu << <grid, block >> > (dev_a, dev_b, dev_c, dev_d, size);
    cudaMemcpy(check, dev_d, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaDeviceReset();

    printf("Host\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", host_d[i]);
    }
    printf("\n");
    printf("Device\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", check[i]);
    }
    printf("\n");

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_d);
    cudaFree(host_a);
    cudaFree(host_b);
    cudaFree(host_c);
    cudaFree(host_d);
    cudaFree(check);

    return 0;
}
