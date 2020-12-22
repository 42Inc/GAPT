#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define N (1024 * 1024)
#define FULL_DATA_SIZE (N * 10)

int main() {
    srand(time(NULL));

    int *dev_a;
    int *dev_a_p;
    int *h_a, *h_b;
    int *h_a_p, *h_b_p;

    float elapsed_time;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_a = (int *) malloc(FULL_DATA_SIZE * sizeof(int));
    h_b = (int *) malloc(FULL_DATA_SIZE * sizeof(int));

    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        h_a[i] = rand() % 1000;
    }

    cudaMalloc((void **) &dev_a, FULL_DATA_SIZE * sizeof(int));

    cudaEventRecord(start, 0);
    cudaMemcpy(dev_a, h_a, FULL_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Elapsed time copy common memory (host->device): %f\n", elapsed_time);

    cudaEventRecord(start, 0);
    cudaMemcpy(h_b, dev_a, FULL_DATA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Elapsed time copy common memory (device->host): %f\n", elapsed_time);

    cudaHostAlloc((void **) &h_a_p, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **) &h_b_p, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

    cudaMalloc((void **) &dev_a_p, FULL_DATA_SIZE * sizeof(int));

    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        h_a_p[i] = rand() % 1000;
    }

    cudaEventRecord(start, 0);
    cudaMemcpy(dev_a_p, h_a_p, FULL_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Elapsed time copy paged-locked memory (host->device): %f\n", elapsed_time);

    cudaEventRecord(start, 0);
    cudaMemcpy(h_b_p, dev_a_p, FULL_DATA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Elapsed time copy paged-locked memory (device->host): %f\n", elapsed_time);

    cudaFree(dev_a);
    cudaFree(dev_a_p);
    cudaFreeHost(h_a_p);
    cudaFreeHost(h_b_p);
    free(h_a);
    free(h_b);

    return 0;
}