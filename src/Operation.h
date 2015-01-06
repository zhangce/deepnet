#ifndef _OPERATION_H
#define _OPERATION_H

#include <assert.h>
#include <math.h>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>


using namespace std;

#define show(x) cerr << #x << " : " << x << endl;

//TODO CHECK AND CHANGE GEMM to cblas_sgemm

#define BLASFUNC(FUNC) FUNC##_

float STEPSIZE = 0.01;
//float INITWEIGHT = 0.1;

inline float logadd(float lna, float lnb)
{
    if (lna == 1.0)
        return lnb;
    if (lnb == 1.0)
        return lna;
    
    float diff = lna - lnb;
    if (diff < 500.0)
        return log(exp(diff) + 1.0) + lnb;
    else
        return lna;
}

class Operation{
public:

	int * groundtruth;

	int mini_batch_size;
	int ninput_feature_map;
	int noutput_feature_map;

	int nrow_output;
	int ncol_output;

	int nrow_input;
	int ncol_input;

	int nrow_conv;
	int ncol_conv;

	float **** inputs;
	float **** grads;
	float **** output;
	float * buf_out;

	float **** grad;
	float * buf_grad;

	float **** weights;
	float * buf_weight;

	virtual void forward(){
		assert(false);
	}

	virtual void backward(){
		assert(false);
	}

	virtual void clear_grad(){
		assert(false);
	}

	Operation(int _mini_batch_size, int _ninput_feature_map, int _noutput_feature_map, 
				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input){
		mini_batch_size=_mini_batch_size;
		ninput_feature_map=_ninput_feature_map;
		noutput_feature_map=_noutput_feature_map;
		nrow_output = _nrow_output;
		ncol_output = _ncol_output;
		nrow_input = _nrow_input;
		ncol_input = _ncol_input;

		groundtruth= new int [_mini_batch_size];
		
		buf_out = new float[mini_batch_size* noutput_feature_map * nrow_output*ncol_output];
		output= new float ***[mini_batch_size];
		for(int mb=0; mb<mini_batch_size; mb++){
			output[mb]=new float **[noutput_feature_map];
			for(int fm=0; fm<noutput_feature_map; fm++){
				output[mb][fm] = new float*[nrow_output];
				for(int r=0;r<nrow_output;r++){
					output[mb][fm][r] = &buf_out[mb*noutput_feature_map*nrow_output*ncol_output+
												fm*nrow_output*ncol_output+
												r*ncol_output];
				}
			}
		}

		buf_grad = new float[mini_batch_size * noutput_feature_map * nrow_output*ncol_output];
		grad= new float ***[mini_batch_size];
		for(int mb=0; mb<mini_batch_size; mb++){
			grad[mb]=new float **[noutput_feature_map];
			for(int fm=0; fm<noutput_feature_map; fm++){
				grad[mb][fm] = new float*[nrow_output];
				for(int r=0;r<nrow_output;r++){
					grad[mb][fm][r] = &buf_grad[mb*noutput_feature_map*nrow_output*ncol_output+
												fm*nrow_output*ncol_output+
												r*ncol_output];
				}
			}
		}

		nrow_conv = nrow_input - nrow_output + 1;
		ncol_conv = ncol_input - ncol_output + 1;
		buf_weight = new float[noutput_feature_map * ninput_feature_map * nrow_conv*ncol_conv];
		weights= new float ***[noutput_feature_map];
		for(int ofm=0; ofm<noutput_feature_map; ofm++){
			weights[ofm]=new float **[ninput_feature_map];
			for(int ifm=0; ifm<ninput_feature_map; ifm++){
				weights[ofm][ifm] = new float*[nrow_conv];
				for(int r=0;r<nrow_conv;r++){
					weights[ofm][ifm][r] = &buf_weight[ofm*ninput_feature_map*nrow_conv*ncol_conv+
												ifm*nrow_conv*ncol_conv+
												r*ncol_conv];
				}
			}
		}
	}

	// Operation(bool isfull, int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input){
	// 	nrow_output = _nrow_output;
	// 	ncol_output = _ncol_output;
	// 	nrow_input = _nrow_input;
	// 	ncol_input = _ncol_input;
		
	// 	_buf = new float[nrow_output*ncol_output];
	// 	output = new float*[nrow_output];
	// 	for(int i=0;i<nrow_output;i++){
	// 		output[i] = &_buf[i*ncol_output];
	// 	}

	// 	_buf_grad = new float[nrow_output*ncol_output];
	// 	grad = new float*[nrow_output];
	// 	for(int i=0;i<nrow_output;i++){
	// 		grad[i] = &_buf_grad[i*ncol_output];
	// 	}
	// }

};

class ConvOperation : public Operation{
public:

	float bias;

	void clear_grad(){
		if(grads[0] != NULL)
			for(int mb=0; mb<mini_batch_size; mb++)
				for(int fm=0; fm<ninput_feature_map; fm++)
					for(int r=0;r<nrow_input;r++)
						for(int c=0; c<ncol_input;c++)
							grads[mb][fm][r][c] = 0;
	}


	ConvOperation(int _mini_batch_size, int _ninput_feature_map, int _noutput_feature_map, 
				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input):
		Operation(_mini_batch_size, _ninput_feature_map, _noutput_feature_map,
				_nrow_output,_ncol_output,_nrow_input,_ncol_input){
	
		// init weights
		for(int ofm=0; ofm<noutput_feature_map; ofm++)
			for(int ifm=0; ifm<ninput_feature_map; ifm++)
				for(int r=0;r<nrow_conv;r++)
					for(int c=0; c<ncol_conv;c++)
						weights[ofm][ifm][r][c] = (drand48()*2-1)/10;

		bias = (drand48()*2-1)/10;

	}

	void backward(){
		for(int mb=0; mb<mini_batch_size; mb++)
			for(int ofm=0; ofm<noutput_feature_map; ofm++)
				for(int r=0;r<nrow_output;r++){
					for(int c=0;c<ncol_output;c++){
						float cvalue = output[mb][ofm][r][c];
						float cgrad = grad[mb][ofm][r][c];

						for(int ifm=0;ifm<ninput_feature_map;ifm++){
							for(int ir=r;ir<r+nrow_conv;ir++){
								for(int ic=c;ic<c+ncol_conv;ic++){

									float w = weights[ofm][ifm][ir-r][ic-c];
									float grad_x = (1.0-cvalue*cvalue)*w * cgrad;

									if(grads[0] != NULL){
										grads[mb][ifm][ir][ic] += grad_x;
									}
								}
							}
						}
					}
				}
		for(int mb=0; mb<mini_batch_size; mb++)
			for(int ofm=0; ofm<noutput_feature_map; ofm++)
				for(int r=0;r<nrow_output;r++){
					for(int c=0;c<ncol_output;c++){
						float cvalue = output[mb][ofm][r][c];
						float cgrad = grad[mb][ofm][r][c];

						for(int ifm=0;ifm<ninput_feature_map;ifm++){
							for(int ir=r;ir<r+nrow_conv;ir++){
								for(int ic=c;ic<c+ncol_conv;ic++){

									float x = inputs[mb][ifm][ir][ic];
									float grad_w = (1.0-cvalue*cvalue)*x * cgrad;
									weights[ofm][ifm][ir-r][ic-c] = 
										weights[ofm][ifm][ir-r][ic-c] + STEPSIZE * grad_w;
								}
							}
						}

						float x = 1.0;
						float grad_w = (1.0-cvalue*cvalue)*x * cgrad;
						bias = bias + STEPSIZE * grad_w;
					}
				}
	}

	void forward(){
		int DSIZE = nrow_input;
		int KSIZE = nrow_conv;
		int I = ninput_feature_map;
		int O = noutput_feature_map;
		int MINIBATCHSIZE = mini_batch_size;

		
		int NDATAROW = I*KSIZE*KSIZE;
		int NDATACOL = (DSIZE-KSIZE+1) * (DSIZE-KSIZE+1) * MINIBATCHSIZE;
		int NKERNELROW = O;

		float * data;
		float alpha[] = {1.0, 1.0};
		float beta [] = {0.0, 0.0};
		char trans='N';

		if (( data = (float *)malloc(sizeof(float) * NDATAROW * NDATACOL)) == NULL){
			fprintf(stderr,"Out of Memory!!\n");exit(1);
		}
		
		// create data_lowered matrix
		int indx=0;
		for(int mb=0; mb<MINIBATCHSIZE; mb++)
			for(int i=0; i<DSIZE-KSIZE+1; i++)
				for(int j=0; j<DSIZE-KSIZE+1; j++)
					for(int f=0; f<I; f++)
						for(int r=0; r<KSIZE; r++)
							for(int s=0; s<KSIZE; s++)
								data[indx++]=inputs[mb][f][i+r][j+s];


		BLASFUNC(sgemm) (&trans, &trans, &O, &NDATACOL, &NDATAROW, alpha, buf_weight, &O, data, &NDATAROW, beta, buf_out, &O );
		//TODO: Add bias
	}


};

class FullyConnectedOperation : public Operation{
public:

	float bias;

	void clear_grad(){
		if(grads[0] != NULL)
			for(int mb=0; mb<mini_batch_size; mb++)
				for(int fm=0; fm<ninput_feature_map; fm++)
					for(int r=0;r<nrow_input;r++)
						for(int c=0; c<ncol_input;c++)
							grads[mb][fm][r][c] = 0;
	}


	FullyConnectedOperation(int _mini_batch_size, int _ninput_feature_map, int _noutput_feature_map, 
				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input):
		Operation(_mini_batch_size, _ninput_feature_map, _noutput_feature_map,
				_nrow_output,_ncol_output,_nrow_input,_ncol_input){
	
		// init weights
		for(int ofm=0; ofm<noutput_feature_map; ofm++)
			for(int ifm=0; ifm<ninput_feature_map; ifm++)
				for(int r=0;r<nrow_conv;r++)
					for(int c=0; c<ncol_conv;c++)
						weights[ofm][ifm][r][c] = (drand48()*2-1)/10;

		bias = (drand48()*2-1)/10;

	}

	void backward(){
		for(int mb=0; mb<mini_batch_size; mb++)
			for(int ofm=0; ofm<noutput_feature_map; ofm++)
				for(int r=0;r<nrow_output;r++){
					for(int c=0;c<ncol_output;c++){
						float cvalue = output[mb][ofm][r][c];
						float cgrad = grad[mb][ofm][r][c];

						for(int ifm=0;ifm<ninput_feature_map;ifm++){
							for(int ir=r;ir<r+nrow_conv;ir++){
								for(int ic=c;ic<c+ncol_conv;ic++){

									float w = weights[ofm][ifm][ir-r][ic-c];
									float grad_x = w * cgrad;

									if(grads[0] != NULL){
										grads[mb][ifm][ir][ic] += grad_x;
									}
								}
							}
						}
					}
				}
		for(int mb=0; mb<mini_batch_size; mb++)
			for(int ofm=0; ofm<noutput_feature_map; ofm++)
				for(int r=0;r<nrow_output;r++){
					for(int c=0;c<ncol_output;c++){
						float cvalue = output[mb][ofm][r][c];
						float cgrad = grad[mb][ofm][r][c];

						for(int ifm=0;ifm<ninput_feature_map;ifm++){
							for(int ir=r;ir<r+nrow_conv;ir++){
								for(int ic=c;ic<c+ncol_conv;ic++){

									float x = inputs[mb][ifm][ir][ic];
									float grad_w = (1.0-cvalue*cvalue)*x * cgrad;
									weights[ofm][ifm][ir-r][ic-c] = 
										weights[ofm][ifm][ir-r][ic-c] + STEPSIZE * grad_w;
								}
							}
						}

						float x = 1.0;
						float grad_w = (1.0-cvalue*cvalue)*x * cgrad;
						bias = bias + STEPSIZE * grad_w;
					}
				}
	}

	void forward(){
		int DSIZE = nrow_input;
		int KSIZE = nrow_conv;
		int I = ninput_feature_map;
		int O = noutput_feature_map;
		int MINIBATCHSIZE = mini_batch_size;

		
		int NDATAROW = I*KSIZE*KSIZE;
		int NDATACOL = (DSIZE-KSIZE+1) * (DSIZE-KSIZE+1) * MINIBATCHSIZE;
		int NKERNELROW = O;

		float * data;
		float alpha[] = {1.0, 1.0};
		float beta [] = {0.0, 0.0};
		char trans='N';

		if (( data = (float *)malloc(sizeof(float) * NDATAROW * NDATACOL)) == NULL){
			fprintf(stderr,"Out of Memory!!\n");exit(1);
		}
		
		// create data_lowered matrix
		int indx=0;
		for(int mb=0; mb<MINIBATCHSIZE; mb++)
			for(int i=0; i<DSIZE-KSIZE+1; i++)
				for(int j=0; j<DSIZE-KSIZE+1; j++)
					for(int f=0; f<I; f++)
						for(int r=0; r<KSIZE; r++)
							for(int s=0; s<KSIZE; s++)
								data[indx++]=inputs[mb][f][i+r][j+s];


		BLASFUNC(sgemm) (&trans, &trans, &O, &NDATACOL, &NDATAROW, alpha, buf_weight, &O, data, &NDATAROW, beta, buf_out, &O );
		//TODO: Add bias
	}
};


class MaxPoolingOperation : public Operation{
public:

	void clear_grad(){
		if(grads[0] != NULL)
			for(int mb=0; mb<mini_batch_size; mb++)
				for(int fm=0; fm<ninput_feature_map; fm++)
					for(int r=0;r<nrow_input;r++)
						for(int c=0; c<ncol_input;c++)
							grads[mb][fm][r][c] = 0;
	}


	MaxPoolingOperation(int _mini_batch_size, int _ninput_feature_map, int _noutput_feature_map, 
				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input):
		Operation(_mini_batch_size, _ninput_feature_map, _noutput_feature_map,
				_nrow_output,_ncol_output,_nrow_input,_ncol_input){
	

		assert(nrow_input % nrow_output == 0);
		assert(ncol_input % ncol_output == 0);
			// TODO: NEED WORK

	}

	void backward(){
			// TODO: FLOAT (== check)
		int row_ratio = nrow_input/nrow_output;
		int col_ratio = ncol_input/ncol_output;
		for(int mb=0; mb<mini_batch_size; mb++)
			for(int ofm=0; ofm<noutput_feature_map; ofm++)
				for(int r=0;r<nrow_output;r++){
					for(int c=0;c<ncol_output;c++){
						float cvalue = output[mb][ofm][r][c];
						float cgrad = grad[mb][ofm][r][c];
						for(int ifm=0; ifm<ninput_feature_map; ifm++){
							for(int ir=r*row_ratio;ir<r*row_ratio+row_ratio;ir++){
								for(int ic=c*col_ratio;ic<c*col_ratio+col_ratio;ic++){
									if(inputs[mb][ifm][ir][ic] == cvalue){
										grads[mb][ifm][ir][ic] += cgrad; 	// TODO: how about if there are two inputs == cvalue 
									}else{
										grads[mb][ifm][ir][ic] = 0;
									}
								}
							}
						}
					}
				}

	}

	void forward(){
		int row_ratio = nrow_input/nrow_output;
		int col_ratio = ncol_input/ncol_output;
		for(int mb=0; mb<mini_batch_size; mb++)
			for(int ofm=0; ofm<noutput_feature_map; ofm++){
				for(int r=0;r<nrow_output;r++){
					for(int c=0;c<ncol_output;c++){
						float max = -10000;
						for(int ifm=0; ifm<ninput_feature_map; ifm++){
							for(int ir=r*row_ratio;ir<r*row_ratio+row_ratio;ir++){
								for(int ic=c*col_ratio;ic<c*col_ratio+col_ratio;ic++){
									if(inputs[mb][0][ir][ic] > max){
										max = inputs[mb][ifm][ir][ic];
									}
								}
							}
						}
						output[mb][ofm][r][c] = max;
					}
				}
			}
	}

};


class SoftmaxOperation : public Operation{
public:


	float*** softweights;
	float* biases;
	int n_label;
	int n_input;

	void clear_grad(){
		if(grads[0] != NULL)
			for(int mb=0; mb<mini_batch_size; mb++)
				for(int fm=0; fm<ninput_feature_map; fm++)
					for(int r=0;r<nrow_input;r++)
						for(int c=0; c<ncol_input;c++)
							grads[mb][fm][r][c] = 0;
	}
	SoftmaxOperation(int _mini_batch_size, int _ninput_feature_map, int _noutput_feature_map, 
				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input):
		Operation(_mini_batch_size, _ninput_feature_map, _noutput_feature_map,
				_nrow_output,_ncol_output,_nrow_input,_ncol_input){
		n_input = _ninput_feature_map;
		n_label = _noutput_feature_map;

		assert(nrow_input == 1);
		assert(ncol_input == 1);
		assert(nrow_output == 1);
		assert(ncol_output == 1);

		biases = new float[n_label];
		for(int i=0;i<n_label;i++){
			biases[i] = 0;
		}
		softweights = new float ** [mini_batch_size];
		for(int mb=0; mb<mini_batch_size; mb++){
			softweights[mb] = new float*[n_label];
			for(int i=0;i<n_label;i++){
				softweights[mb][i] = new float[n_input];
				for(int j=0;j<n_input;j++){
					softweights[mb][i][j] = (drand48()*2-1)/10;
				}
			}
		}
	}

	void backward(){
		for(int mb=0; mb<mini_batch_size; mb++)
			for(int label=0;label<n_label;label++){
				float cvalue = output[mb][label][0][0];
				for(int i_input=0;i_input<n_input;i_input++){

					float w = softweights[mb][label][i_input];
					float x = inputs[mb][i_input][0][0];

					float grad_w = (label == groundtruth[mb])*x - cvalue*x;
					float grad_x = (label == groundtruth[mb])*w - cvalue*w;

					softweights[mb][label][i_input] = 
						softweights[mb][label][i_input] + STEPSIZE * grad_w;

					grads[mb][i_input][0][0] += grad_x;

				}

				float w = biases[label];
				float x = 1.0;
				float grad_w = (label == groundtruth[mb])*x - cvalue*x;
				float grad_x = (label == groundtruth[mb])*w - cvalue*w;
				biases[label] = biases[label] + STEPSIZE * grad_w;
			}
	}

	void forward(){
		for(int mb=0; mb<mini_batch_size; mb++)
			for(int i=0;i<n_label;i++){
				float sum = 0.0;
				for(int i_input=0;i_input<n_input;i_input++){
					sum += softweights[mb][i][i_input] * inputs[mb][i_input][0][0];
				}
				sum += biases[i];
				output[mb][i][0][0] = sum;
			}

		for(int mb=0; mb<mini_batch_size; mb++){
			float sum = -100000;
			for(int i=0;i<n_label;i++){
				sum = logadd(sum, output[mb][i][0][0]); 
			}
			for(int i=0;i<n_label;i++){
				output[mb][i][0][0] = exp(output[mb][i][0][0]-sum);
			}
		}
	}

};

class RELUOperation : public Operation{
public:

	void clear_grad(){
		if(grads[0] != NULL)
			for(int mb=0; mb<mini_batch_size; mb++)
				for(int fm=0; fm<ninput_feature_map; fm++)
					for(int r=0;r<nrow_input;r++)
						for(int c=0; c<ncol_input;c++)
							grads[mb][fm][r][c] = 0;
	}


	RELUOperation(int _mini_batch_size, int _ninput_feature_map, int _noutput_feature_map, 
				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input):
		Operation(_mini_batch_size, _ninput_feature_map, _noutput_feature_map,
				_nrow_output,_ncol_output,_nrow_input,_ncol_input){
			;
	}

	void backward(){
		for(int mb=0; mb<mini_batch_size; mb++)
			for(int ofm=0; ofm<noutput_feature_map; ofm++)
				for(int r=0;r<nrow_output;r++){
					for(int c=0;c<ncol_output;c++){
						if(output[mb][ofm][r][c]>0)
							grads[mb][ofm][r][c]=grad[mb][ofm][r][c];
					}
				}

	}

	void forward(){
		for(int mb=0; mb<mini_batch_size; mb++)
			for(int fm=0; fm<noutput_feature_map; fm++)
				for(int r=0;r<nrow_output;r++)
					for(int c=0;c<ncol_output;c++){
					   output[mb][fm][r][c] = std::max(inputs[mb][fm][r][c], (float)0.0);
					}
	}

};

class DataOperation : public Operation{
public:

	float bias;

	void clear_grad(){
		;
	}

	DataOperation(int _mini_batch_size, int _ninput_feature_map, int _noutput_feature_map, 
				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input):
		Operation(_mini_batch_size, _ninput_feature_map, _noutput_feature_map,
				_nrow_output,_ncol_output,_nrow_input,_ncol_input){
			;
	}

	void backward(){
		;
	}

	void forward(){
		;
	}


};



#endif










