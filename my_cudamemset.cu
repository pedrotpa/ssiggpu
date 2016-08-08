#include "my_cudamemset.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<typename T>
__global__ void initKernel(T * devPtr, const T val, const size_t nwords)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (; tidx < nwords; tidx += stride)
		devPtr[tidx] = val;
}

extern "C"
void my_cudamemset(float *(&d_devPtr), size_t size, const float val){
	int blockSize = 768;
	int gridSize = 4;
	gridSize = (size + blockSize - 1) / blockSize;
	initKernel<float><<<gridSize, blockSize >> >(d_devPtr, val, size);
	cudaDeviceSynchronize();
}