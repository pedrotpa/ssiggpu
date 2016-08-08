#ifndef H_MY_CUDAMEMSET
#define H_MY_CUDAMEMSET

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C"
void my_cudamemset(float *(&d_devPtr), size_t size, const float val);

#endif
