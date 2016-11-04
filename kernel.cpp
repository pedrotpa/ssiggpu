#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cublas_v2.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>

#include <opencv2\opencv.hpp>

#include "errchk.h"
#include "timer.h"
#include "dsv.h"
#include "simPLS.cuh"

#define epsilion_e 0.0000001

inline int IDX2C(int i, int j, int ld){
	return (j*ld) + i;
}

uint32_t seedi = 50;
uint32_t rngi(){
	seedi ^= seedi << 13;
	seedi ^= seedi >> 17;
	seedi ^= seedi << 5;
	return seedi;
}

uint32_t rngi(uint32_t seed){
	seedi = seed;
	return rngi();
}

uint32_t seedf = 51;
float rngf(){
	seedf ^= seedf << 13;
	seedf ^= seedf >> 17;
	seedf ^= seedf << 5;
	return (float)seedf / (float)0xffffffff;
}
float rngf(uint32_t seed){
	seedf = seed;
	return rngf();
}

void rnginput(float *(&X0), float *(&Y0), int n, int m=4,uint32_t si=50, uint32_t sf=51){
	rngi(si);
	rngf(sf);
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			X0[IDX2C(i, j, n)] = (rngi() % (int)(n / m)) + rngf();
			Y0[IDX2C(i, j, n)] = (rngi() % (int)(n / m)) + rngf();
		}
	}
	return;
}

void debug(cublasHandle_t handle, const std::string s, float *d_a, int Nrows, int Ncols){
	printf("%s:\n", s.c_str());
	float *h_debug = (float *)malloc(Nrows*Ncols*sizeof(float));
	int tmp = Ncols == 1 ? Nrows : Nrows == 1 ? Ncols : -1;
	if (tmp != -1){
		cublasGetVector(tmp, sizeof(float), d_a, 1, h_debug, 1);
	}
	else{
		cublasGetMatrix(Nrows, Ncols, sizeof(float), d_a, Nrows, h_debug, Nrows);
	}
	if (tmp == -1){
		for (int i = 0; i < Nrows; i++){
			for (int j = 0; j < Ncols; j++){
				printf("%4.4f  ", h_debug[IDX2C(i, j, Nrows)]);
			}
			printf("\n");
		}
	}
	else{
		for (int i = 0; i < tmp; i++){
			printf("%4.4f ", h_debug[i]);
		}
	}
	free(h_debug);
	printf("\n");
}

void deb(const std::string s, float *a, int Nrows, int Ncols){
	printf("%s:\n", s.c_str());
	int tmp = Ncols == 1 ? Nrows : Nrows == 1 ? Ncols : -1;
	if (tmp == -1){
		for (int i = 0; i < Nrows; i++){
			for (int j = 0; j < Ncols; j++){
				printf("%3.4f\t", a[IDX2C(i, j, Nrows)]);
			}
			printf("\n");
		}
	}
	else{
		for (int i = 0; i < tmp; i++){
			printf("%4.4f\t", a[i]);
		}
	}
	printf("\n");
}

void teste_dsv(float *(&h_X0), float *(&h_Y0), int n){
	deb("X0", h_X0, n, n);
	deb("Y0", h_Y0, n, n);
	cublasHandle_t handle;
	cublasSafeCall(cublasCreate(&handle));

	float *d_X0;			gpuErrchk(cudaMalloc(&d_X0, n*n*sizeof(float)));
	float *d_Y0;			gpuErrchk(cudaMalloc(&d_Y0, n*n*sizeof(float)));

	cublasSafeCall(cublasSetMatrix(n, n, sizeof(float), h_X0, n, d_X0, n));									//copy to d_X0
	cublasSafeCall(cublasSetMatrix(n, n, sizeof(float), h_Y0, n, d_Y0, n));									//copy to d_Y0

	float *d_v1;			gpuErrchk(cudaMalloc(&d_v1, n*sizeof(float)));									//Nrows
	float *d_v2;			gpuErrchk(cudaMalloc(&d_v2, n*sizeof(float)));									//Ncols

	std::cout << "lambda: " << dsv(handle, d_X0, n, n, d_v1, d_v2, 1 / 100000) << std::endl;
	debug(handle, "U1", d_v1, n, 1);
	debug(handle, "V1", d_v2, n, 1);
}

void bench(int n, int ncomp_max,int n_samples,int comp_init){
	float *X0 = (float *)malloc(n*n*sizeof(float));
	float *Y0 = (float *)malloc(n*n*sizeof(float));
	rnginput(X0, Y0, n, n / 32, 50, 51);

	int dx = n; int dy = n;

	for (int ncomp = comp_init; ncomp <= ncomp_max; ncomp++){
		for (int i = 0; i < n_samples; i++){
			float *Xloadings, *Yloadings, *Xscores, *Yscores, *Weights, *beta;
			double cpu0 = get_wall_time();
			simPLS(n, dx, dy, X0, Y0, ncomp, Xloadings, Yloadings, Xscores, Yscores, Weights, beta);
			double cpu1 = get_wall_time();
			//printf("%4.6f\t", cpu1 - cpu0);
			deb("Yloadings", Yloadings, n, ncomp);

			exit(0);
			free(Xloadings);
			free(Yloadings);
			free(Xscores);
			free(Yscores);
			free(Weights);
			free(beta);
		}
		std::cout << std::endl;
	}
	free(X0);
	free(Y0);
}

void speedup(int step, int n_step, int ncomp, int n_samples) {
	for (int i = 1; i <= n_step;i++){
		int n = i*step;
		float *X0 = (float *)malloc(n*n*sizeof(float));
		float *Y0 = (float *)malloc(n*n*sizeof(float));
		rnginput(X0, Y0, n, n / 32, 50, 51);

		int dx = n; int dy = n;
		for (int j = 0; j < n_samples; j++) {
			float *Xloadings, *Yloadings, *Xscores, *Yscores, *Weights, *beta;
			double cpu0 = get_wall_time();
			simPLS(n, dx, dy, X0, Y0, ncomp, Xloadings, Yloadings, Xscores, Yscores, Weights, beta);
			double cpu1 = get_wall_time();
			printf("%4.6f\t", cpu1 - cpu0);
			free(Xloadings);
			free(Yloadings);
			free(Xscores);
			free(Yscores);
			free(Weights);
			free(beta);
		}
		printf("\n");
		free(X0);
		free(Y0);
	}
}

int main(int argc, char *argv[]){
	int step, n_step, ncomp, n_samples;
	sscanf(argv[1], "%d", &step);
	sscanf(argv[2], "%d", &n_step);
	sscanf(argv[3], "%d", &ncomp);
	sscanf(argv[4], "%d", &n_samples);
	std::cout << "step: " << step << "\tn_step: " << n_step << "\tncomp: " << ncomp << "\tn_samples: " << n_samples << std::endl;
	//ex speedup(200, 20, 4, 10);
	speedup(step,n_step,ncomp, n_samples);
	
	return 0;


	int n;
	int comp_init;
	//int ncomp;
	//int n_samples;
	/*sscanf(argv[1], "%d", &n);
	sscanf(argv[2], "%d", &ncomp);
	sscanf(argv[3], "%d", &n_samples);
	if (argc == 5)
		sscanf(argv[4], "%d", &comp_init);
	else{
		comp_init = 1;
	}
	bench(n,ncomp,n_samples,comp_init);
	return 0;*/
	if (argc != 3 && argc != 5){
		std::cout << "simPLS n ncomp seedi seedf" << std::endl;
		return 1;
	}
	//int n;
	//int ncomp;
	sscanf(argv[1], "%d", &n);
	sscanf(argv[2], "%d", &ncomp);
	if (argc == 5){
		sscanf(argv[3], "%d", &seedi);
		sscanf(argv[4], "%d", &seedf);
	}

	float *X0 = (float *)malloc(n*n*sizeof(float));
	float *Y0 = (float *)malloc(n*n*sizeof(float));
	rnginput(X0, Y0, n, n / 32, 50, 51);

	int dx = n; int dy = n;
	float *Xloadings, *Yloadings, *Xscores, *Yscores, *Weights, *beta;
	double cpu0 = get_cpu_time();
	simPLS(n, dx, dy, X0, Y0, ncomp, Xloadings, Yloadings, Xscores, Yscores, Weights, beta);
	double cpu1 = get_cpu_time();
	std::cout << n << "\t" << cpu1 - cpu0 << std::endl;
	free(X0);
	free(Y0);
	return 0;
	char *fname;
	sprintf(fname, "cuda%d-%d.yml", n, ncomp);
	cv::FileStorage fs(fname, cv::FileStorage::WRITE);
	cv::Mat tmp = cv::Mat(dx, ncomp, CV_32F, Xloadings);
	fs << "Xloadings" << tmp;
	tmp = cv::Mat(dy, ncomp, CV_32F, Yloadings);
	fs << "Yloadings" << tmp;
	tmp = cv::Mat(n, ncomp, CV_32F, Xscores);
	fs << "Xscores" << tmp;
	tmp = cv::Mat(n, ncomp, CV_32F, Yscores);
	fs << "Yscores" << tmp;
	tmp = cv::Mat(dx, ncomp, CV_32F, Weights);
	fs << "Weights" << tmp;
	tmp = cv::Mat(dx + 1, dy, CV_32F, beta);
	fs << "beta" << tmp;
	fs.release();

	return EXIT_SUCCESS;
}
