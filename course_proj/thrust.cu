#include <iostream>
#include <cmath>
#include <functional>
#include <random>
#include <sys/time.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/generate.h>
#define N 4

double wtime()
{
    struct timeval t;
    gettimeofday (&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

float initMatr2()
{
    std::default_random_engine generator;
    generator.seed(std::random_device()());
    std::uniform_real_distribution<float> distrib(0, 100);
    return distrib(generator);
}

void thrustCuda()
{
    thrust::host_vector<float> mA(N);
    thrust::host_vector<float> mB(N);
    thrust::generate(mA.begin(), mA.end(), initMatr2);
    thrust::generate(mB.begin(), mB.end(), initMatr2);
    thrust::device_vector<float> mA_d(mA);
    thrust::device_vector<float> mB_d(mB);
    thrust::device_vector<float> mC_d(mB);

    for (int i = 0; i < N; i++)
        thrust::transform(mA_d.begin(), mA_d.end(), mB_d.begin(), 
mC_d.begin(), thrust::plus<float>());
    thrust::host_vector<float> mC(mC_d);
    for (int i = 0; i < N; i++)
        std::cout << mA[i] << " ";
    std::cout << std::endl;
    for (int i = 0; i < N; i++)
        std::cout << mB[i] << " ";
    std::cout << std::endl;
    for (int i = 0; i < N; i++)
        std::cout << mC[i] << " ";
    std::cout << std::endl;
}

int main()
{
    double time = -wtime();
    thrustCuda();
    time += wtime();
    std::cout << time << std::endl;
    return 0;
 
}

