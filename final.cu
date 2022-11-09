#include <vector>
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <stdlib.h> 
#include <chrono>

using namespace std;
using namespace std::chrono;

#define N 1000000
#define SM_SIZE 4096


/*
global functions:
1. random number generator, generating the normal variables
2. monte calo simulation, calculating the payoffs for each path and if the path expired in or out of the money
3. sum of payment and calculate the final option value (achieved via parallel reduction algorithm) 
*/

// kernel for calculating the simulated final option values
__global__ void FV_kernel(
    float * d_normal,
    float * d_result, 
    float * d_s,
    float S,
    float T, 
    float K, 
    float r, 
    float v
    )
{
    // defining the thread id in global memory
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const float S_adjust = S * exp(T*(r-0.5*v*v));
    /*
	parallel part, each one of the 1M thread takes care of
    one pricing process, returning an array of simulated final 
	option prices
    */ 
    if (idx < N)
    {
        d_result[idx] = S_adjust * exp(v*sqrt(T)*d_normal[idx]);
	// determining if the option expires ITM or OTM
        float payoff = (d_result[idx] > K ? d_result[idx] - K : 0.f);
	d_s[idx] = payoff; 
    }
}



// kernel for parallel reduction
__global__ void parallel_reduction(float * d_in, float * d_out)
{
    unsigned idx =  threadIdx.x + blockIdx.x * blockDim.x;
    unsigned tid =  threadIdx.x; 
    // using shared memory for each threadblock to accelerate the summing process
    __shared__ float shared_mem[SM_SIZE];
    // loading from global to shared memory to perform option value summing process
    shared_mem[tid] = d_in[idx];
    __syncthreads();
    /*
	strided memory access to prevent bank conflicts, first round of reduction
	reduces each element in
     */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
	if (threadIdx.x < s)
	{
	    shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
	}
        __syncthreads();
    }
    
    if (threadIdx.x == 0)
    {
	d_out[blockIdx.x] = shared_mem[0];
    
    }
}


// overloaded kernel for parallel reduction and discounting the option to PV
__global__ void parallel_reduction(float * d_in, float * d_out, float r, float T)
{
    unsigned idx =  threadIdx.x + blockIdx.x * blockDim.x;
    unsigned tid =  threadIdx.x; 
    __shared__ float shared_mem[SM_SIZE];
    shared_mem[tid] = d_in[idx];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
	if (threadIdx.x < s)
	{
	    shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
	}
        __syncthreads();
    }
    
    if (threadIdx.x == 0)
    {
	d_out[blockIdx.x] = shared_mem[0]*exp(-r*T)/N;
    }
}



int main(int argc, char *argv[]) 
{
    /*
	Initializing the variables on CPU, and allocating the array spaces on CPU
    */

    float S1 = 90.0f;
    float S2 = 100.0f;
    float S3 = 110.0f;
    float T = 1.0f; 
    float K = 100.0f; 
    float r = 0.03f;
    float v = 0.3f; 
    float * d_normal; 
    float * d_result;
    float * d_s;
    float * d_sum;
    float * h_sum_1 = (float *)malloc(sizeof(float));
    float * h_sum_2 = (float *)malloc(sizeof(float));
    float * h_sum_3 = (float *)malloc(sizeof(float));
    // threadblock size and grid size
    int n_thread = 1024;
    int n_block = ceil(N/n_thread); 
  
    // allocating memory on device
    cudaMalloc(&d_normal, N * sizeof(float)); 
    cudaMalloc(&d_result, N * sizeof(float));
    cudaMalloc(&d_s, N * sizeof(float));
    cudaMalloc(&d_sum, N * sizeof(float));

    
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    // generate the random values
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32); 
    // setting seed
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    // generate N normally distributed variables
    curandGenerateNormal(gen, d_normal, N, 0.0f, 1.0f);
    cudaDeviceSynchronize();

    /*
	Timing the performance for pricing 3 options
        First time running the applications
    */
 
    // option1 
    FV_kernel<<<n_block, n_thread>>>(d_normal, d_result, d_s, S1, T, K, r, v);
    parallel_reduction<<<n_block, n_thread>>>(d_s, d_sum);
    // after the first round of reduction sum, discount the options to PV
    parallel_reduction<<<1, n_thread>>>(d_sum, d_sum, r, T);
    cudaMemcpy(h_sum_1, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    float price1 = h_sum_1[0];

    // option2
    FV_kernel<<<n_block, n_thread>>>(d_normal, d_result, d_s, S2, T, K, r, v);
    parallel_reduction<<<n_block, n_thread>>>(d_s, d_sum);
    parallel_reduction<<<1, n_thread>>>(d_sum, d_sum, r, T);
    cudaMemcpy(h_sum_2, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    float price2 = h_sum_2[0];

    // option3
    FV_kernel<<<n_block, n_thread>>>(d_normal, d_result, d_s, S3, T, K, r, v);
    parallel_reduction<<<n_block, n_thread>>>(d_s, d_sum);
    parallel_reduction<<<1, n_thread>>>(d_sum, d_sum, r, T);
    cudaMemcpy(h_sum_3, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    float price3 = h_sum_3[0];
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    std::cout << "Total elapsed time: " << duration_cast<milliseconds>(t2 - t1).count() << " ms" << std::endl;
    std::cout << std::endl; 

    /*
	Running the codes for a second time (as per the assignment requirement, 
        and printing out the option values
    */

    // option1 
    FV_kernel<<<n_block, n_thread>>>(d_normal, d_result, d_s, S1, T, K, r, v);
    parallel_reduction<<<n_block, n_thread>>>(d_s, d_sum);
    // after the first round of reduction sum, discount the options to PV
    parallel_reduction<<<1, n_thread>>>(d_sum, d_sum, r, T);
    cudaMemcpy(h_sum_1, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    price1 = h_sum_1[0];

    // option2
    FV_kernel<<<n_block, n_thread>>>(d_normal, d_result, d_s, S2, T, K, r, v);
    parallel_reduction<<<n_block, n_thread>>>(d_s, d_sum);
    parallel_reduction<<<1, n_thread>>>(d_sum, d_sum, r, T);
    cudaMemcpy(h_sum_2, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    price2 = h_sum_2[0];

    // option3
    FV_kernel<<<n_block, n_thread>>>(d_normal, d_result, d_s, S3, T, K, r, v);
    parallel_reduction<<<n_block, n_thread>>>(d_s, d_sum);
    parallel_reduction<<<1, n_thread>>>(d_sum, d_sum, r, T);
    cudaMemcpy(h_sum_3, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    price3 = h_sum_3[0];
    

    std::cout << "European call with S = " << S1 << ", K = 100 , r = 0.03, v = 0.3, T = 1 and N = 1000000 has price of " << price1 << endl;
    std::cout << "European call with S = " << S2 << ", K = 100 , r = 0.03, v = 0.3, T = 1 and N = 1000000 has price of " << price2 << endl;
    std::cout << "European call with S = " << S3 << ", K = 100 , r = 0.03, v = 0.3, T = 1 and N = 1000000 has price of " << price3 << endl;


    // deallocating the GPU memories and objects
    curandDestroyGenerator(gen);
    cudaFree(d_s);
    cudaFree(d_result);
    cudaFree(d_normal);
    cudaFree(d_sum);

}

 



