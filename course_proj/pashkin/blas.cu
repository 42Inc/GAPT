#include <iostream>
#include <functional>
#include <random>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#define N 4

double wtime()
{
    struct timeval t;
    gettimeofday (&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
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

void cuBlas()
{
    auto mA = new float[N];
    auto mB = new float[N];
    auto mC = new float[N];
    initVect(mA);
    initVect(mB);

    float *mA_d,
          *mB_d;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMalloc((void **)&mA_d, N * sizeof(float));
    cudaMalloc((void **)&mB_d, N * sizeof(float));
    cublasSetVector(N, sizeof(float), mA, 1, mA_d, 1);
    cublasSetVector(N, sizeof(float), mB, 1, mB_d, 1);
    float mul = 1.0f;
    cublasSaxpy(handle, N, &mul, mA_d, 1, mB_d, 1);
    cublasGetVector(N, sizeof(float), mB_d, 1, mC, 1);

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
    delete[] mA;
    delete[] mB;
    delete[] mC;
}

int main()
{
    double time = -wtime();
    cuBlas();
    time += wtime();
    std::cout << time << std::endl;

    return 0;
}

