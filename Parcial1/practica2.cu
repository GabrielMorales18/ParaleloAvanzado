
//practica 1
/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void idx_calc_tid(int *input)
{
    int tid = threadIdx.x;
    printf("[DEVICE] threadIdx.x: %d, data: %d\n\r", tid, input[tid]);
}

int main()
{
    const int N = 16;
    const int array[N] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int size = N * sizeof(int);

    int* d_a = 0;

    cudaMalloc((void**)&d_a, size);

    cudaMemcpy(d_a, array, size, cudaMemcpyHostToDevice);

    idx_calc_tid << <1, N >> > (d_a);

    cudaDeviceSynchronize();

    cudaDeviceReset();

    cudaFree(d_a);

    return 0;
}
*/

//practica 2
/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void idx_calc_tid(int *input)
{
    int tid = threadIdx.x;
    printf("[DEVICE] threadIdx.x: %d, data: %d\n\r", tid, input[tid]);
}

int main()
{
    const int N = 16;
    const int array[N] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int size = N * sizeof(int);

    int* d_a = 0;

    cudaMalloc((void**)&d_a, size);

    cudaMemcpy(d_a, array, size, cudaMemcpyHostToDevice);

    idx_calc_tid << <2, 8 >> > (d_a);

    cudaDeviceSynchronize();

    cudaDeviceReset();

    cudaFree(d_a);

    return 0;
}
*/

//practica 3
/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void idx_calc_gid(int *input)
{
    int tid = threadIdx.x;
    //gid = tid + offset
    int offset = blockIdx.x * blockDim.x;
    int gid = tid + offset;
    printf("[DEVICE] blockIdx.x: %d, threadIdx.x: %d, gid: %d,  data: %d\n\r", blockIdx.x, tid, gid, input[gid]);
}

int main()
{
    const int N = 16;
    const int array[N] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int size = N * sizeof(int);

    int* d_a = 0;

    cudaMalloc((void**)&d_a, size);

    cudaMemcpy(d_a, array, size, cudaMemcpyHostToDevice);

    idx_calc_gid << <4, 4 >> > (d_a);

    cudaDeviceSynchronize();

    cudaDeviceReset();

    cudaFree(d_a);

    return 0;
}
*/

//practica 4
/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void idx_calc_gid(int *input)
{
    int tid = threadIdx.x;
    //gid = tid + offset
    int offsetBlock = blockIdx.x * blockDim.x;
    int offsetRow = blockIdx.y * gridDim.x * blockDim.x;
    int gid = tid + offsetBlock + offsetRow;
    printf("[DEVICE]  gridDim.x: %d, blockIdx.x: %d, threadIdx.x: %d, blockIdx.y: %d, gid: %d,  data: %d\n\r", gridDim.x, blockIdx.x, tid, blockIdx.y, gid, input[gid]);
}

int main()
{
    const int N = 16;
    const int array[N] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int size = N * sizeof(int);

    int* d_a = 0;

    cudaMalloc((void**)&d_a, size);

    cudaMemcpy(d_a, array, size, cudaMemcpyHostToDevice);

    dim3 grid(2, 2);
    dim3 block(4);

    idx_calc_gid << <grid, block >> > (d_a);

    cudaDeviceSynchronize();

    cudaDeviceReset();

    cudaFree(d_a);

    return 0;
}
*/

//practica 5

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void idx_calc_gid(int *input)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    //gid = tid + offset
    int offsetBlock = blockIdx.x * blockDim.x * blockDim.y;
    int offsetRow = blockIdx.y * gridDim.x * blockDim.x * blockDim.y;
    int gid = tid + offsetBlock + offsetRow;
    printf("[DEVICE]  gridDim.x: %d, blockIdx.x: %d, threadIdx.x: %d,  blockIdx.y: %d, gid: %d,  data: %d\n\r", gridDim.x, blockIdx.x, tid, blockIdx.y, gid, input[gid]);
}

int main()
{
    const int N = 16;
    const int array[N] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int size = N * sizeof(int);

    int* d_a = 0;

    cudaMalloc((void**)&d_a, size);

    cudaMemcpy(d_a, array, size, cudaMemcpyHostToDevice);

    dim3 grid(2, 2);
    dim3 block(2, 2);

    idx_calc_gid << <grid, block >> > (d_a);

    cudaDeviceSynchronize();

    cudaDeviceReset();

    cudaFree(d_a);

    return 0;
}
	
