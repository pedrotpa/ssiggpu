#include "dsv.h"

float dsv(cublasHandle_t handle, float *d_A, int Nrows, int Ncols, float *(&d_Um), float *(&d_Vm), float epsilon){
	float *d_aat;					gpuErrchk(cudaMalloc(&d_aat, Nrows*Nrows*sizeof(float)));
	float *d_ata;					gpuErrchk(cudaMalloc(&d_ata, Ncols*Ncols*sizeof(float)));

	float al = 1.0f;
	float bet = 0.0f;
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, Nrows, Nrows, Ncols, &al, d_A, Nrows, d_A, Nrows, &bet, d_aat, Nrows);
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Ncols, Ncols, Nrows, &al, d_A, Nrows, d_A, Nrows, &bet, d_ata, Ncols);

	int mv = Nrows > Ncols ? Nrows : Ncols;
	float *h_ones_max = (float *)malloc(mv*sizeof(float));
	for (int i = 0; i < mv; i++) h_ones_max[i] = 1.0f;

	if (Nrows > Ncols){
		cublasSafeCall(cublasSetVector(Nrows, sizeof(float), h_ones_max, 1, d_Um, 1));
		cublasSafeCall(cublasScopy(handle, Ncols, d_Um, 1, d_Vm, 1));
	}
	else{
		cublasSafeCall(cublasSetVector(Ncols, sizeof(float), h_ones_max, 1, d_Vm, 1));
		cublasSafeCall(cublasScopy(handle, Nrows, d_Vm, 1, d_Um, 1));
	}

	power_m(handle, d_aat, Nrows, epsilon, STARTING_EIGENVALUE, d_Um);
	float lambda = power_m(handle, d_ata, Ncols, epsilon, STARTING_EIGENVALUE, d_Vm);
	cudaFree(d_aat);
	cudaFree(d_ata);
	free(h_ones_max);
	return sqrt(lambda);
}

float power_m(cublasHandle_t handle, float *d_a, int m, float epsilon, float mu, float *d_x){
	float dd = 1.0f;
	float n = mu;		//10.0f
	float *d_y;			gpuErrchk(cudaMalloc(&d_y, m*sizeof(float)));
	float al;
	float bet;
	float result;
	while (dd > epsilon){
		al = 1.0f;
		bet = 0.0f;
		cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, m, m, &al, d_a, m, d_x, 1, &bet, d_y, 1));
		cublasSafeCall(cublasSnrm2(handle, m, d_x, 1, &result));
		dd = fabs(result - n);
		n = result;
		al = 1 / n;
		cublasSafeCall(cublasScopy(handle, m, d_y, 1, d_x, 1));
		cublasSafeCall(cublasSscal(handle, m, &al, d_x, 1));
	}
	cublasSafeCall(cublasSnrm2(handle, m, d_x, 1, &result));
	al = 1 / result;
	cublasSafeCall(cublasSscal(handle, m, &al, d_x, 1));
	cudaFree(d_y);
	return n;
}
