#include <iostream>
#include <cmath>
#include <functional>
#include <random>
#include <cuda.h>
#include <sys/time.h>
#define  N 4

const int threadsPerBlock = 1024;
const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

double wtime()
{
    struct timeval t;
    gettimeofday (&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

__global__ void vpv(float *mA_d, float *mB_d, float *mC_d)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = i;
    if (tid < N)

    mC_d[i] = mA_d[i] + mB_d[i];
}

void initVect(float *vector)
{
    std::default_random_engine generator;
    generator.seed(std::random_device()());
    std::uniform_real_distribution<double> distrib(0, 100);
    auto getRand = std::bind(distrib, generator);

    for (int i = 0; i < N; i++)
        vector[i] = getRand();
}

void brutCuda()
{
    auto mA = new float[N];
    auto mB = new float[N];
    auto mC = new float[N];
    initVect(mA);
    initVect(mB);
    initVect(mC);
    float *mA_d,
          *mB_d,
          *mC_d;
    //int threadsPerBlock = 1024;
    //int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    cudaMalloc((void **)&mA_d, N * sizeof(float));
    cudaMalloc((void **)&mB_d, N * sizeof(float));
    cudaMalloc((void **)&mC_d, N * sizeof(float));
    cudaMemcpy(mA_d, mA, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mB_d, mB, N * sizeof(float), cudaMemcpyHostToDevice);

    vpv <<< blocksPerGrid, threadsPerBlock >>> (mA_d, mB_d, mC_d);
    cudaMemcpy(mC, mC_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        std::cout << mA[i] << " ";
    std::cout << std::endl;
    for (int i = 0; i < N; i++)
        std::cout << mB[i] << " ";
    std::cout << std::endl;
    for (int i = 0; i < N; i++) 
        std::cout << mC[i] << " ";
    std::cout << std::endl;

    cudaFree(mA_d);
    cudaFree(mB_d);
    cudaFree(mC_d);
    delete[] mA;
    delete[] mB;
    delete[] mC;
}

int main()
{
    double time = -wtime();
    brutCuda();
    time += wtime();
    std::cout << time << std::endl;
    return 0;
}

