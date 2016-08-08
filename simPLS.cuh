#ifndef H_SIMPLS
#define H_SIMPLS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cublas_v2.h"

#include "errchk.h"
#include "dsv.h"
#include "my_cudamemset.cuh"

#define epsilion_e 0.0000001

inline void simPLS(const int n, const int dx, const int dy, const float *X0, const float *Y0, const int ncomp, float *(&Xloadings), float *(&Yloadings));
inline void simPLS(const int n, const int dx, const int dy, const float *X0, const float *Y0, const int ncomp, float *(&Xloadings), float *(&Yloadings),
	float *(&Xscores), float *(&Yscores));
inline void simPLS(const int n, const int dx, const int dy, const float *X0, const float *Y0, const int ncomp, float *(&Xloadings), float *(&Yloadings),
	float *(&Xscores), float *(&Yscores), float *(&Weights), float *(&beta));

void simPLS_f(const int n, const int dx, const int dy, const float *h_X0, const float *h_Y0, const int ncomp, float *(&h_Xloadings), float *(&h_Yloadings),
	float *(&h_Xscores), float *(&h_Yscores), float *(&h_Weights), float *(&h_beta), const int nargout);



void simPLS_f(const int n, const int dx, const int dy, const float *h_X0, const float *h_Y0, const int ncomp, float *(&h_Xloadings), float *(&h_Yloadings),
	float *(&h_Xscores), float *(&h_Yscores), float *(&h_Weights), float *(&h_beta), const int nargout){
	//h_X0 = N x DX
	//h_Y0 = N x DY
	cublasHandle_t handle;
	cublasSafeCall(cublasCreate(&handle));


	// --- output variables
	float *d_Yloadings;
	float *d_Xscores;
	float *d_Yscores;
	float *d_Weights;
	float *d_beta;

	float *d_X0;			gpuErrchk(cudaMalloc(&d_X0, n*dx*sizeof(float)));
	float *d_Y0;			gpuErrchk(cudaMalloc(&d_Y0, n*dy*sizeof(float)));
	float *d_n_ones;		gpuErrchk(cudaMalloc(&d_n_ones, n*sizeof(float)));
	float *d_meanX;			gpuErrchk(cudaMalloc(&d_meanX, dx*sizeof(float)));
	float *d_meanY;			gpuErrchk(cudaMalloc(&d_meanY, dy*sizeof(float)));

	cublasSafeCall(cublasSetMatrix(n, dx, sizeof(float), h_X0, n, d_X0, n));									//copy to d_X0
	cublasSafeCall(cublasSetMatrix(n, dy, sizeof(float), h_Y0, n, d_Y0, n));									//copy to d_Y0
	my_cudamemset(d_n_ones, n, 1.0f);

	float al = 1.0f / (float)n;
	float bet = 0.0f;
	cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_T, n, dx, &al, d_X0, n, d_n_ones, 1, &bet, d_meanX, 1));		//=meanX0
	cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_T, n, dy, &al, d_Y0, n, d_n_ones, 1, &bet, d_meanY, 1));		//=meanY0

	al = -1.0;
	cublasSafeCall(cublasSger(handle, n, dx, &al, d_n_ones, 1, d_meanX, 1, d_X0, n));							//X0 = bsxfun(@minus, X0, meanX)
	cublasSafeCall(cublasSger(handle, n, dy, &al, d_n_ones, 1, d_meanY, 1, d_Y0, n));							//Y0 = bsxfun(@minus, Y0, meanY)
	cudaFree(d_n_ones);

	// --- Preallocate outputs
	h_Xloadings = (float *)malloc(dx*ncomp*sizeof(float));
	h_Yloadings = (float *)malloc(dy*ncomp*sizeof(float));	gpuErrchk(cudaMalloc(&d_Yloadings, dy*ncomp*sizeof(float)));
	if (nargout > 2){
		h_Xscores = (float *)malloc(n*ncomp*sizeof(float));	gpuErrchk(cudaMalloc(&d_Xscores, n*ncomp*sizeof(float)));
		h_Yscores = (float *)malloc(dy*ncomp*sizeof(float)); gpuErrchk(cudaMalloc(&d_Yscores, n*ncomp*sizeof(float)));
		if (nargout > 4){
			h_Weights = (float *)malloc(n*ncomp*sizeof(float)); gpuErrchk(cudaMalloc(&d_Weights, dx*ncomp*sizeof(float)));
			h_beta = (float *)malloc((dx + 1)*dy*sizeof(float));	gpuErrchk(cudaMalloc(&d_beta, (dx + 1)*dy*sizeof(float)));
		}
	}

	float *d_cov;			 gpuErrchk(cudaMalloc(&d_cov, dx*dy*sizeof(float)));
	al = 1.0f; bet = 0.0f;
	cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, dx, dy, n, &al, d_X0, n, d_Y0, n, &bet, d_cov, dx));//cov=X0'*Y0

	// --- SVD results space
	float *d_ri;            gpuErrchk(cudaMalloc(&d_ri, dx*sizeof(float)));
	float si;
	float *d_ci;			gpuErrchk(cudaMalloc(&d_ci, dy*sizeof(float)));


	//preallocate variables
	float *d_ti;			gpuErrchk(cudaMalloc(&d_ti, n*sizeof(float)));
	float *d_vi;			gpuErrchk(cudaMalloc(&d_vi, dx*sizeof(float)));
	float *d_vj;			gpuErrchk(cudaMalloc(&d_vj, dx*sizeof(float)));
	float *d_V;				gpuErrchk(cudaMalloc(&d_V, dx*ncomp*sizeof(float)));
	float *d_detmp;			gpuErrchk(cudaMalloc(&d_detmp, dy*ncomp*sizeof(float)));

	/*
	* An orthonormal basis for the span of the X loadings, to make the successive
	* deflation X0'*Y0 simple - each new basis vector can be removed from Cov
	* separately.
	*/
	cudaMemset(d_V, 0, dx*ncomp*sizeof(float));

	for (int i = 1; i <= ncomp; i++){
		/*
		* Find unit length ti=X0*ri and ui=Y0*ci whose covariance, ri'*X0'*Y0*ci, is
		* jointly maximized, subject to ti'*tj=0 for j=1:(i-1).
		*/
		si = dsv(handle, d_cov, dx, dy, d_ri, d_ci, epsilion_e);

		al = 1.0f;		bet = 0.0f;
		cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, n, dx, &al, d_X0, n, d_ri, 1, &bet, d_ti, 1));					//=ti

		float normti;		cublasSafeCall(cublasSnrm2(handle, n, d_ti, 1, &normti));									//=normti
		al = 1.0f / normti;
		cublasSafeCall(cublasSscal(handle, n, &al, d_ti, 1));															//=ti/normti

		al = 1.0f;
		cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_T, n, dx, &al, d_X0, n, d_ti, 1, &bet, d_vi, 1));					//vi=X0'*ti
		cublasSafeCall(cublasGetVector(dx, sizeof(float), d_vi, 1, &h_Xloadings[(i - 1)*dx], 1));						//Xloadings(:,i)=vi

		al = si / normti;
		cublasSafeCall(cublasSscal(handle, dy, &al, d_ci, 1));															//qi=si*ci/normti
		cublasSafeCall(cublasScopy(handle, dy, d_ci, 1, d_Yloadings+(dy*(i - 1)), 1));									//Yloadings(:,i)=qi

		if (nargout > 2){
			cublasSafeCall(cublasScopy(handle, n, d_ti, 1, d_Xscores+(n*(i - 1)), 1));									//Xscores(:,i)=ti
			al = 1.0; bet = 0.0;
			cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, n, dy, &al, d_Y0, n, d_ci, 1, &bet, d_Yscores + (i - 1)*dy, 1));
			if (nargout > 4){
				al = 1.0f / normti;
				cublasSafeCall(cublasSaxpy(handle, dx, &al, d_ri, 1, d_Weights + (i - 1)*dy, 1));
			}
		}

		/*
		* Update the orthonormal basis with modified Gram Schmidt (more stable),
		* repeated twice (ditto).
		*/
		for (int repeat = 1; repeat <= 2; repeat++){
			for (int j = 1; j <= i - 1; j++){
				cublasSafeCall(cublasScopy(handle, dx, d_V + (j - 1)*dx, 1, d_vj, 1));									//vj=V(:,j)
				float result;
				cublasSafeCall(cublasSdot(handle, dx, d_vi, 1, d_vj, 1, &result));										//
				result = -result;																						//	vi=vi - (vi'*vj)*vj;
				cublasSafeCall(cublasSaxpy(handle, dx, &result, d_vj, 1, d_vi, 1));										//
			}
		}

		float normvi;
		cublasSafeCall(cublasSnrm2(handle, dx, d_vi, 1, &normvi));														//
		al = 1.0 / normvi;																								//	vi=vi/norm(vi);
		cublasSafeCall(cublasSscal(handle, dx, &al, d_vi, 1));															//
		cublasSafeCall(cublasScopy(handle, dx, d_vi, 1, d_V+(dx*(i - 1)), 1));											//	V(:,i)=vi;

		/*
		* Deflate Cov, i.e. project onto the ortho-complement of the X loadings.
		* First remove projections along the current basis vector, then remove any
		* component along previous basis vectors that's crept in as noise from
		* previous deflations.
		*/

		al = 1.0; bet = 0.0;
		cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_T, dx, dy, &al, d_cov, dx, d_vi, 1, &bet, d_detmp, 1));				//d_detmp=vi'*Cov
		al = -1.0;
		cublasSafeCall(cublasSger(handle, dx, dy, &al, d_vi, 1, d_detmp, 1, d_cov, dx));									//cov=cov-vi*detmp

		al = 1.0; bet = 0.0;
		cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, i, dy, dx, &al, d_V, dx, d_cov, dx, &bet, d_detmp, i));//d_detmp=Vi'*Cov
		al = -1.0; bet = 1.0;
		cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dx, dy, i, &al, d_V, dx, d_detmp, i, &bet, d_cov, dx));//cov=cov-Vi*detmp
	}
	cudaFree(d_X0);
	cudaFree(d_Y0);
	cudaFree(d_cov);

	cudaFree(d_ri);
	cudaFree(d_ci);

	cudaFree(d_ti);
	cudaFree(d_vi);
	cudaFree(d_vj);
	cudaFree(d_V);
	cudaFree(d_detmp);

	cublasSafeCall(cublasGetMatrix(dy, ncomp, sizeof(float), d_Yloadings, dy, h_Yloadings, dy));
	if (nargout > 2){
		cublasSafeCall(cublasGetMatrix(n, ncomp, sizeof(float), d_Xscores, n, h_Xscores, n));
		if (nargout > 4){
			cublasSafeCall(cublasGetMatrix(dx, ncomp, sizeof(float), d_Weights, dx, h_Weights, dx));
		}
	}
	if (nargout > 2){
		/*
		* By convention, orthogonalize the Y scores w.r.t. the preceding Xscores,
		* i.e. XSCORES'*YSCORES will be lower triangular.  This gives, in effect, only
		* the "new" contribution to the Y scores for each PLS component.  It is also
		* consistent with the PLS-1/PLS-2 algorithms, where the Y scores are computed
		* as linear combinations of a successively-deflated Y0.  Use modified
		* Gram-Schmidt, repeated twice.
		*/
		for (int i = 1; i <= ncomp; i++){
			float *d_ui = d_Yscores + (n*(i - 1));
			for (int repeat = 1; repeat <= 2; repeat++){
				for (int j = 1; j <= i - 1; j++){
					float *d_tj = d_Xscores + (n*(j - 1));
					float result;
					cublasSafeCall(cublasSdot(handle, n, d_ui, 1, d_tj, 1, &result));
					result = -result;
					cublasSafeCall(cublasSaxpy(handle, n, &result, d_tj, 1, d_ui, 1));
				}
			}
		}
		cublasSafeCall(cublasGetMatrix(n, ncomp, sizeof(float), d_Yscores, n, h_Yscores, n));
		cudaFree(d_Xscores);
		cudaFree(d_Yscores);

		if (nargout > 4){
			al = 1.0;
			bet = 0.0;
			cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, dy, dx, ncomp, &al, d_Yloadings, dy, d_Weights, dx, &bet, d_beta + dy, dy));	//Yloadings*Weights'=beta(:,2:dy)'
			cublasSafeCall(cublasScopy(handle, dy, d_meanY, 1, d_beta, 1)); //beta(:,1)=meanY
			al = -1.0; bet = 1.0;
			cublasSgemv(handle, CUBLAS_OP_N, dy, dx, &al, d_beta + dy, dy, d_meanX, 1, &bet, d_beta, 1);	//beta(:,1)=beta(:,1)-meanX*beta(:,2:dy)'

			cublasSafeCall(cublasGetMatrix(dx + 1, dy, sizeof(float), d_beta, dx + 1, h_beta, dx + 1));
			cudaFree(d_beta);
			cudaFree(d_Weights);
		}
	}
	cudaFree(d_Yloadings);
	cudaFree(d_meanX);
	cudaFree(d_meanY);
	cublasDestroy(handle);
}

inline void simPLS(const int n, const int dx, const int dy, const float *X0, const float *Y0, const int ncomp, float *(&Xloadings), float *(&Yloadings)){
	float *null_ptr = NULL;
	simPLS_f(n, dx, dy, X0, Y0, ncomp, Xloadings, Yloadings, null_ptr, null_ptr, null_ptr, null_ptr, 2);
}
inline void simPLS(const int n, const int dx, const int dy, const float *X0, const float *Y0, const int ncomp, float *(&Xloadings), float *(&Yloadings),
	float *(&Xscores), float *(&Yscores)){
	float *null_ptr = NULL;
	simPLS_f(n, dx, dy, X0, Y0, ncomp, Xloadings, Yloadings, Xscores, Yscores, null_ptr, null_ptr, 4);
}
inline void simPLS(const int n, const int dx, const int dy, const float *X0, const float *Y0, const int ncomp, float *(&Xloadings), float *(&Yloadings),
	float *(&Xscores), float *(&Yscores), float *(&Weights), float *(&beta)){
	simPLS_f(n, dx, dy, X0, Y0, ncomp, Xloadings, Yloadings, Xscores, Yscores, Weights, beta, 6);
}


#endif
