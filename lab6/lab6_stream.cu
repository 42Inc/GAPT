#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define N (896 * 896)
#define FULL_DATA_SIZE (1024 * 1024 * 10)

__global__ void add_vectors(int *a, int *b, int *c) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void mul_vectors(int *a, int *b, int *c) { 
    __shared__ float cache[256];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    int temp = 0;

    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (cacheIndex < s) {
            cache[cacheIndex] += cache[cacheIndex + s];
        }
        __syncthreads();
    }

    if (cacheIndex == 0) c[blockIdx.x] = cache[0];
}

int main() {
    srand(time(NULL));

    float elapsed_time;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStream_t stream_0, stream_1;
    cudaStreamCreate(&stream_0);
    cudaStreamCreate(&stream_1);

    int *h_a_p, *h_b_p, *h_c_p;
    int *dev_a0, *dev_b0, *dev_c0, *dev_a1, *dev_b1, *dev_c1;

    cudaHostAlloc((void **) &h_a_p, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **) &h_b_p, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **) &h_c_p, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaMalloc((void **)&dev_a0, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void **)&dev_b0, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void **)&dev_c0, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void **)&dev_a1, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void **)&dev_b1, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void **)&dev_c1, FULL_DATA_SIZE * sizeof(int));
    
    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        h_a_p[i] = rand() % 1000;
        h_b_p[i] = rand() % 1000;
    }

    cudaEventRecord(start, 0);

    for (int i = 0; i < FULL_DATA_SIZE; i += N * 2) {
        cudaMemcpyAsync(dev_a0, h_a_p + i, N * sizeof(int), cudaMemcpyHostToDevice, stream_0);
        cudaMemcpyAsync(dev_a1, h_a_p + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream_1);
        
        cudaMemcpyAsync(dev_b0, h_b_p + i, N * sizeof(int), cudaMemcpyHostToDevice, stream_0);
        cudaMemcpyAsync(dev_b1, h_b_p + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream_1);

        add_vectors <<< N / 256, 256, 0, stream_0 >>> (dev_a0, dev_b0, dev_c0);
        add_vectors <<< N / 256, 256, 0, stream_1 >>> (dev_a1, dev_b1, dev_c1);

        cudaMemcpyAsync(h_c_p + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream_0);
        cudaMemcpyAsync(h_c_p + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream_1);
    }

    cudaStreamSynchronize(stream_0);
    cudaStreamSynchronize(stream_1);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Elapsed time (add vectors): %f\n", elapsed_time);

    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        h_c_p[i] = 0;
    }
    
    cudaEventRecord(start, 0);

    for (int i = 0; i < FULL_DATA_SIZE; i += N * 2) {
        cudaMemcpyAsync(dev_a0, h_a_p + i, N * sizeof(int), cudaMemcpyHostToDevice, stream_0);
        cudaMemcpyAsync(dev_a1, h_a_p + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream_1);

        cudaMemcpyAsync(dev_b0, h_b_p + i, N * sizeof(int), cudaMemcpyHostToDevice, stream_0);
        cudaMemcpyAsync(dev_b1, h_b_p + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream_1);

        mul_vectors <<< N / 256, 256, 0, stream_0 >>> (dev_a0, dev_b0, dev_c0);
        mul_vectors <<< N / 256, 256, 0, stream_1 >>> (dev_a1, dev_b1, dev_c1);

        cudaMemcpyAsync(h_c_p + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream_0);
        cudaMemcpyAsync(h_c_p + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream_1);
    }
    cudaStreamSynchronize(stream_0);
    cudaStreamSynchronize(stream_1);

    long long result = 0;
    for (int i = 0; i < (N / 256); i++) result += h_c_p[i];

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("Elapsed time (mul vectors): %f\n", elapsed_time);

    cudaFree(dev_a0);
    cudaFree(dev_a1);
    cudaFree(dev_b0);
    cudaFree(dev_b1);
    cudaFree(dev_c0);
    cudaFree(dev_c1);
    cudaFreeHost(h_a_p);
    cudaFreeHost(h_b_p);
    cudaFreeHost(h_c_p);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream_0);
    cudaStreamDestroy(stream_1);

    return 0;
}