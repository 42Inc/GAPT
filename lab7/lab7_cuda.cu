#include <stdio.h>

__global__ void initFun(int *nf) {
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    nf[n] *= 10;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "USAGE: main <num_of_devices> " 
                   "<device_indices>\n");
        return -1;
    }

    int *info_devs = (int *) calloc(argc - 2, sizeof(int));

    info_devs[0] = atoi(argv[1]);
    for (int i = 1; i < argc - 2; i++) {
        info_devs[i] = atoi(argv[i + 1]);
    }

    int N = atoi(argv[4]);
    float elapsed_time;

    //printf("%d\n", N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStream_t* streams;
    int **nfd = (int **) calloc(info_devs[0], sizeof(int *));
    int **nfh = (int **) calloc(info_devs[0], sizeof(int *));

    streams = (cudaStream_t *) calloc(info_devs[0], sizeof(cudaStream_t));

    cudaEventRecord(start, 0);

    for (int i = 0; i < info_devs[0]; i++) {
        cudaSetDevice(info_devs[i + 1]);
        cudaStreamCreate(&streams[i]);

        cudaMalloc((void **) &nfd[i], (N / info_devs[0]) * sizeof(int));
        cudaMallocHost((void **) &nfh[i], (N / info_devs[0]) * sizeof(int));
     
        for (int n = 0; n < N / info_devs[0]; n++)
            nfh[i][n] = n + i * N / info_devs[0];

        cudaMemcpyAsync(nfd[i], nfh[i], (N / info_devs[0]) * sizeof(int),
                    cudaMemcpyHostToDevice, streams[i]);
       
        initFun <<< N / info_devs[0] / 32, 32, 0, streams[i] >>>(nfd[i]);

        cudaMemcpyAsync(nfh[i], nfd[i], (N / info_devs[0]) * sizeof(int),
                    cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < info_devs[0]; i++) {
        cudaSetDevice(info_devs[i + 1]);
        cudaStreamSynchronize(streams[i]);
        
        //for (int n = 0; n < N / info_devs[0]; n++)
            //fprintf(stderr, "nfh[%d][%d] = %d\n", i, n, nfh[i][n]);
    }

    cudaSetDevice(info_devs[1]);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("%f\n", elapsed_time);

    for (int i = 0; i < info_devs[0]; i++) {   
        cudaStreamDestroy(streams[i]);
        cudaFree(nfd[i]);
        cudaFreeHost(nfh[i]);
        cudaDeviceReset();
    }

    return 0;
}