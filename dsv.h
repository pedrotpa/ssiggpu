#ifndef DSV_H
#define DSV_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cublas_v2.h"

#include "errchk.h"

float power_m(cublasHandle_t handle, float *d_a, int m, float epsilon, float mu, float *d_x);

#define STARTING_EIGENVALUE 10.0f

float dsv(cublasHandle_t handle, float *d_A, int Nrows, int Ncols, float *(&d_Um), float *(&d_Vm), float epsilon);

#endif