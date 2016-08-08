#ifndef ERRCHK_H
#define ERRCHK_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>

/**********************/
/* cuBLAS ERROR CHECK */
/**********************/
#ifndef cublasSafeCall
#define cublasSafeCall(err)     __cublasSafeCall(err, __FILE__, __LINE__)
#endif
inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line){
	if (CUBLAS_STATUS_SUCCESS != err){
		fprintf(stderr, "cuBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n", __FILE__, __LINE__, err);
		getchar(); cudaDeviceReset(); exit(err);
	}
}

/********************/
/* CUDA ERROR CHECK */
/********************/
#ifndef gpuErrchk
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#endif
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true){
	if (code != cudaSuccess){
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#endif
