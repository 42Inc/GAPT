#include <stdio.h>
#include <pthread.h>

struct th_args {
    int *nfh;
    int *nfd;
    int dev_num;
};

int N, cnt_dev;

__global__ void initFun(int *nf) {
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    nf[n] *= 10;
}

void *stream(void *args) {
    struct th_args *info_dev = (struct th_args *) args;

    cudaSetDevice(info_dev->dev_num);

    cudaMalloc((void **) &info_dev->nfd, (N / cnt_dev) * sizeof(int));
    cudaMallocHost((void **) &info_dev->nfh, (N / cnt_dev) * sizeof(int));
 
    for (int n = 0; n < N / cnt_dev; n++)
        info_dev->nfh[n] = n + info_dev->dev_num * N / cnt_dev;

    cudaMemcpyAsync(info_dev->nfd, info_dev->nfh, (N / cnt_dev) * sizeof(int),
                cudaMemcpyHostToDevice);
   
    initFun <<< N / cnt_dev / 32, 32, 0 >>>(info_dev->nfd);

    cudaMemcpyAsync(info_dev->nfh, info_dev->nfd, (N / cnt_dev) * sizeof(int),
                cudaMemcpyDeviceToHost);

    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "USAGE: main <num_of_devices>\n");
        return -1;
    }

    cnt_dev = atoi(argv[1]);
    N = atoi(argv[2]);

    float elapsed_time;
    
    //printf("%d\n", N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    pthread_t tid[cnt_dev];
    struct th_args *args = (struct th_args *) calloc(cnt_dev, sizeof(struct th_args));

    for (int i = 0; i < cnt_dev; i++) {
        args[i].dev_num = i;
    }

    cudaEventRecord(start, 0);

    for (int i = 0; i < cnt_dev; i++) {
        pthread_create(&tid[i], NULL, stream, (void *) &args[i]);
    }

    for (int i = 0; i < cnt_dev; i++) {
        pthread_join(tid[i], NULL);
    }

    cudaSetDevice(args[0].dev_num);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("%f\n", elapsed_time);

    /*for (int i = 0; i < cnt_dev; i++) {
        for (int n = 0; n < N / cnt_dev; n++)
            fprintf(stderr, "nfh[%d][%d] = %d\n", i, n, args[i].nfh[n]);
    }*/

    for (int i = 0; i < cnt_dev; i++) {   
        cudaFree(args[i].nfd);
        cudaFreeHost(args[i].nfh);
        cudaDeviceReset();
    }

    return 0;
}