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
#include <random>
#include <pthread.h>

///dgemm("N","N", n, m, k, alpha, B, n, A, k, beta, C, n);
// opA(A) is of dimension m×k and opB(B) is of dimension k×n,

using namespace std;

#define show(x) cerr << #x << " : " << x << endl;
// C :=  alpha * op(A) * op(B) + beta*C,

#define BLASF(FUNC) FUNC##_


float STEPSIZE = 0.01;

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

class Operation;

class node{
public:
	Operation * op;
	int mini_batch_size;
	int starting_ind;
	int type; //0 forward, 1 backward
	
	node(Operation * _op, int _mini_batch_size, int _starting_index,int _type){
		op=_op;
		mini_batch_size=_mini_batch_size;
		starting_ind=_starting_index;
		type=_type;
	}
};

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

	int stride;
	int pad;
	int group;

	void weights_show(){
		cout << "WEIGHTS:" << endl;
		for(int ofm=0; ofm<noutput_feature_map; ofm++){
			for(int ifm=0; ifm<ninput_feature_map; ifm++){
				for(int r=0;r<nrow_conv;r++){
					for(int c=0; c<ncol_conv; c++){
						cout << weights[ofm][ifm][r][c] << " ";
					}
					cout << endl;
				}
				cout << endl;
			}
			cout << "------" << endl;
		}	
	}

	void inputs_show(){
		cout << "INPUTS:" << endl;	
		for(int mb=0; mb<mini_batch_size; mb++){
			for(int fm=0; fm<ninput_feature_map; fm++){
				for(int r=0;r<nrow_input;r++){
					for(int c=0; c<nrow_input; c++){
						cout << inputs[mb][fm][r][c] << " ";
					}
					cout << endl;
				}
				cout << endl;
			}
			cout << "------" << endl;
		}	
	}

	void output_show(){
		cout << "OUTPUTS:" << endl;	
		for(int mb=0; mb<mini_batch_size; mb++){
			for(int fm=0; fm<noutput_feature_map; fm++){
				for(int r=0;r<nrow_output;r++){
					for(int c=0; c<ncol_output; c++){
						cout << output[mb][fm][r][c] << " ";
					}
					cout << endl;
				}
				cout << endl;
			}
			cout << "------" << endl;
		}
	}

	void grads_show(){
		cout << "GRADS:" << endl;	
		for(int mb=0; mb<mini_batch_size; mb++){
			for(int fm=0; fm<ninput_feature_map; fm++){
				for(int r=0;r<nrow_input;r++){
					for(int c=0; c<nrow_input; c++){
						cout << grads[mb][fm][r][c] << " ";
					}
					cout << endl;
				}
				cout << endl;
			}
			cout << "------" << endl;
		}	
	}

	void grad_show(){
		cout << "GRAD:" << endl;	
		for(int mb=0; mb<mini_batch_size; mb++){
			for(int fm=0; fm<noutput_feature_map; fm++){
				for(int r=0;r<nrow_output;r++){
					for(int c=0; c<ncol_output; c++){
						cout << grad[mb][fm][r][c] << " ";
					}
					cout << endl;
				}
				cout << endl;
			}
			cout << "------" << endl;
		}
	}
	   /** Returns true if the thread was successfully started, false if there was an error starting the thread */
	void start_forward_thread(int num_core_per_chunk, int max_core)
	{
		// cerr << "Number of cores per thread: " << num_core_per_chunk << endl;
		int n_threads=max_core/num_core_per_chunk;
		// cerr << "Number of threads: " << n_threads << endl;
		int CSIZE=mini_batch_size/n_threads;

		pthread_t * thread=new pthread_t [n_threads];
		int * iret = new int [n_threads];
		struct timeval start, stop;
		double time1;
		gettimeofday( &start, (struct timezone *)0);
		for(int j=0; j<n_threads; j++){
			int starting_ind=j*CSIZE;
			int * core_ids=new int [num_core_per_chunk];
			for(int k=0; k<num_core_per_chunk; k++)
				core_ids[k]=k+j*num_core_per_chunk;
			node input_node(this,CSIZE,starting_ind,0);
			iret[j] = pthread_create( &thread[j], NULL, InternalThreadEntryFunc, (void*) &input_node);
		}
		for(int j=0; j<n_threads; j++)
			pthread_join( thread[j], NULL);

	}
	void start_backward_thread(int num_core_per_chunk, int max_core)
	{
		// cerr << "Number of cores per thread: " << num_core_per_chunk << endl;
		int n_threads=max_core/num_core_per_chunk;
		// cerr << "Number of threads: " << n_threads << endl;
		int CSIZE=mini_batch_size/n_threads;

		pthread_t * thread=new pthread_t [n_threads];
		int * iret = new int [n_threads];
		struct timeval start, stop;
		double time1;
		gettimeofday( &start, (struct timezone *)0);
		for(int j=0; j<n_threads; j++){
			int starting_ind=j*CSIZE;
			int * core_ids=new int [num_core_per_chunk];
			for(int k=0; k<num_core_per_chunk; k++)
				core_ids[k]=k+j*num_core_per_chunk;
			node input_node(this, CSIZE,starting_ind,1);
			iret[j] = pthread_create( &thread[j], NULL, InternalThreadEntryFunc, (void *) &input_node);
		}
		for(int j=0; j<n_threads; j++)
			pthread_join( thread[j], NULL);

	}


	static void * InternalThreadEntryFunc(void * inp) {
		node * node_temp=(node *)inp;
		int mini_batch_size=node_temp->mini_batch_size;
		int starting_ind=node_temp->starting_ind;
		Operation * This = node_temp->op;
		int type=node_temp->type;
		// show(mini_batch_size);
		// show(starting_ind)
		// show(type)
		if(type==0)
			This->forward(mini_batch_size,starting_ind); 
		else
			This->backward(mini_batch_size,starting_ind);
		return NULL;
	}


	virtual void forward(int batch_core, int starting_ind){
		assert(false);
	}

	virtual void backward(int batch_core, int starting_ind){
		assert(false);
	}

	virtual void clear_grad(){
		assert(false);
	}

	Operation(int _mini_batch_size, int _ninput_feature_map, int _noutput_feature_map, 
				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input, int _stride=1, int _pad=0, int _group=1){
		mini_batch_size=_mini_batch_size;
		ninput_feature_map=_ninput_feature_map;
		noutput_feature_map=_noutput_feature_map;
		nrow_output = _nrow_output;
		ncol_output = _ncol_output;
		nrow_input = _nrow_input;
		ncol_input = _ncol_input;
		stride=_stride;
		pad=_pad;
		group=_group;

		groundtruth= new int [_mini_batch_size];
		// show("A")
		
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
		// show("B")
		// show(mini_batch_size)
		// show(noutput_feature_map)
		// show(nrow_output)
		// show(ncol_output)
		buf_grad = new float[mini_batch_size * noutput_feature_map * nrow_output*ncol_output];
		grad= new float ***[mini_batch_size];
		// show("D")
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
				// show("C")

		nrow_conv = nrow_input+2*pad - (nrow_output - 1) * stride;
		ncol_conv = ncol_input+2*pad - (ncol_output - 1) * stride;
		// show(nrow_conv)
		// show(ncol_conv)
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

};

class ConvOperation : public Operation{
public:

	float * bias;

	void clear_grad(){
		if(grads[0] != NULL)
			for(int mb=0; mb<mini_batch_size; mb++)
				for(int fm=0; fm<ninput_feature_map; fm++)
					for(int r=0;r<nrow_input;r++)
						for(int c=0; c<ncol_input;c++)
							grads[mb][fm][r][c] = 0;
	}


	ConvOperation(int _mini_batch_size, int _ninput_feature_map, int _noutput_feature_map, 
				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input, int _stride=1, int _pad=0, int _group=1):
		Operation(_mini_batch_size, _ninput_feature_map, _noutput_feature_map,
				_nrow_output,_ncol_output,_nrow_input,_ncol_input,_stride,_pad,_group){
		show(_mini_batch_size)
		// init weights
		for(int ofm=0; ofm<noutput_feature_map; ofm++)
			for(int ifm=0; ifm<ninput_feature_map; ifm++)
				for(int r=0;r<nrow_conv;r++)
					for(int c=0; c<ncol_conv;c++)
						weights[ofm][ifm][r][c] = (drand48()*2-1)/10;
		bias = new float [noutput_feature_map];
		for(int ofm=0; ofm<noutput_feature_map; ofm++)
			bias[ofm] = (drand48()*2-1)/10;
	}

	void backward(int batch_core, int starting_ind){	
		blasint KSIZE = nrow_conv;
		blasint DSIZE = nrow_output+2*(KSIZE-1);
		blasint I = noutput_feature_map/group;
		blasint O = ninput_feature_map/group;
		blasint MINIBATCHSIZE = mini_batch_size;
		blasint NCONV=(DSIZE-KSIZE)/stride+1;
		 
		blasint NDATAROW = I*KSIZE*KSIZE;
		blasint NDATACOL = NCONV * NCONV * MINIBATCHSIZE;
		blasint NKERNELROW = O;
		blasint OFFSET_KERN = NKERNELROW * NDATAROW;

		float alpha[] = {1.0, 1.0};
		float beta []= {0.0, 0.0};

		char no_trans = 'N';
		char trans = 'T';

		float * buf_grads;
		float * buf_grad_temp;
		buf_grads = (float *)malloc(sizeof(float) * O * NDATACOL);
		buf_grad_temp = (float *)malloc(sizeof(float) * NDATAROW * NDATACOL); 
		float * weight_diff = (float *)malloc(sizeof(float) * NDATAROW * O);
		float * buf_weight_temp = (float *)malloc(sizeof(float) * I * O * KSIZE*KSIZE);
		int indw=0;
		for(int ofm=0; ofm<I; ofm++)
			for(int ifm=0; ifm<O; ifm++)
				for(int r=KSIZE-1; r>=0; r--)
					for(int c=KSIZE-1; c>=0; c--)
						buf_weight_temp[indw++]=weights[ofm][ifm][r][c];

		// Calculating data gradiant
		for(int g=0;g<group;g++){
			int indx=0;
			for(int mb=0; mb<mini_batch_size; mb++)
				for(int fm=g*I; fm<(g+1)*I; fm++)
					for(int r=0;r<KSIZE;r++)
						for(int c=0; c<KSIZE; c++)
							for(int i=0; i<NCONV*stride; i+=stride)
								for(int j=0; j<NCONV*stride; j+=stride){
									if(i+r<KSIZE-1||i+r>=DSIZE-KSIZE+1||j+c<KSIZE-1||j+c>=DSIZE-KSIZE+1)
										buf_grad_temp[indx++] =0;
									else			
										buf_grad_temp[indx++] = grad[mb][fm][i+r-KSIZE+1][j+c-KSIZE+1];
								}


			BLASF(sgemm) (&no_trans, &no_trans, &NDATACOL, &O, &NDATAROW, alpha, buf_grad_temp, &NDATACOL, buf_weight_temp+g*OFFSET_KERN, &NDATAROW, beta, buf_grads, &NDATACOL);

			int idx =0;        
			for(int mb=0; mb<MINIBATCHSIZE; mb++)
				for(int fm=g*O; fm<(g+1)*O; fm++)
					for(int r=0; r<NCONV; r++)
						for(int c=0; c<NCONV; c++)
							grads[mb][fm][r][c] = buf_grads[idx++];
		}
			
		KSIZE = nrow_output;
		DSIZE = nrow_input+2*pad;
		I = ninput_feature_map/group;
		O = noutput_feature_map/group;
		MINIBATCHSIZE = mini_batch_size;
		NCONV=nrow_conv;

		NDATAROW = KSIZE*KSIZE*MINIBATCHSIZE;
		NDATACOL = NCONV * NCONV * I;// * MINIBATCHSIZE;
		NKERNELROW = O;
		OFFSET_KERN = NKERNELROW * NDATAROW;

		float * data;

		if (( data = (float *)malloc(sizeof(float) * NDATAROW * NDATACOL)) == NULL){
				fprintf(stderr,"Out of Memory!!\n");exit(1);
		}

		// Calculating weight gradiant
		for(int g=0;g<group;g++){
			int indx=0;
			for(int mb=0; mb<MINIBATCHSIZE; mb++)
				for(int r=0; r<KSIZE; r++)
					for(int c=0; c<KSIZE; c++)
						for(int fm=g*I; fm<(g+1)*I; fm++)
							for(int i=0; i<NCONV*stride; i+=stride)
								for(int j=0; j<NCONV*stride; j+=stride){
									if(i+r<pad || i+r>=nrow_input+pad || j+c<pad || j+c>=ncol_input+pad)
										data[indx++]=0;
									else
										data[indx++]=inputs[mb][fm][i+r-pad][j+c-pad];
								}
			BLASF(sgemm) (&no_trans, &no_trans, &NDATACOL, &O, &NDATAROW, alpha, data, &NDATACOL, buf_grad, &NDATAROW, beta, weight_diff, &NDATACOL);

			for(int iter=g*O*NDATACOL;iter<(g+1)*O*NDATACOL;iter++){
				buf_weight[iter] = buf_weight[iter] - STEPSIZE*weight_diff[iter-g*O*NDATACOL];
			}                  								
		}	
	}

	void forward(int batch_core, int starting_ind){
		blasint DSIZE = nrow_input+2*pad;
		blasint KSIZE = nrow_conv;
		blasint I = ninput_feature_map/group;
		blasint O = noutput_feature_map/group;
		blasint MINIBATCHSIZE = mini_batch_size;
		blasint NCONV=(DSIZE-KSIZE)/stride+1;
		
		blasint NDATAROW = I*KSIZE*KSIZE;
		blasint NDATACOL = NCONV * NCONV * MINIBATCHSIZE;
		blasint NKERNELROW = O;
		blasint OFFSET_KERN = NKERNELROW * NDATAROW;

		float * data;
		float alpha[] = {1.0, 1.0};
		float beta []= {0.0, 0.0};

		char trans='N';
		if (( data = (float *)malloc(sizeof(float) * NDATAROW * NDATACOL)) == NULL){
			fprintf(stderr,"Out of Memory!!\n");exit(1);
		}
		float * buf_out_temp= new float [NKERNELROW*NDATACOL];
		for(int g=0; g<group; g++){
			// create data_lowered matrix
			int indx=0;
			for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
				for(int fm=g*I; fm<(g+1)*I; fm++)
					for(int r=0; r<KSIZE; r++)
						for(int c=0; c<KSIZE; c++)
							for(int i=0; i<NCONV*stride; i+=stride)
								for(int j=0; j<NCONV*stride; j+=stride){
									if(i+r<pad || i+r>=nrow_input+pad || j+c<pad || j+c>=ncol_input+pad)
										data[indx++]=0;
									else
										data[indx++]=inputs[mb][fm][i+r-pad][j+c-pad];
									// cout << data[indx-1] << " ";
								}
								// cout << endl;

			BLASF(sgemm) (&trans, &trans, &NDATACOL, &O, &NDATAROW, alpha, data, &NDATACOL, buf_weight+g*OFFSET_KERN, &NDATAROW, beta, buf_out_temp, &NDATACOL );
			
			indx=0;
			for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
				for(int fm=g*O; fm<(g+1)*O; fm++)
					for(int r=0;r<nrow_output;r++)
						for(int c=0; c<ncol_output; c++)
							output[mb][fm][r][c]=buf_out_temp[indx++]+bias[fm];
		}
		

		// for(int i=0; i<O*NDATACOL; i++)
		// 	cout << buf_out[i] << " " ;
		// cout << endl;
	}


};

class FullyConnectedOperation : public Operation{
public:

	float * bias;

	void clear_grad(){
		if(grads[0] != NULL)
			for(int mb=0; mb<mini_batch_size; mb++)
				for(int fm=0; fm<ninput_feature_map; fm++)
					for(int r=0;r<nrow_input;r++)
						for(int c=0; c<ncol_input;c++)
							grads[mb][fm][r][c] = 0;
	}


	FullyConnectedOperation(int _mini_batch_size, int _ninput_feature_map, int _noutput_feature_map, 
				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input, int _stride=1, int _pad=0, int _group=1):
		Operation(_mini_batch_size, _ninput_feature_map, _noutput_feature_map,
				_nrow_output,_ncol_output,_nrow_input,_ncol_input,_stride,_pad,_group){
	
		// init weights
		for(int ofm=0; ofm<noutput_feature_map; ofm++)
			for(int ifm=0; ifm<ninput_feature_map; ifm++)
				for(int r=0;r<nrow_conv;r++)
					for(int c=0; c<ncol_conv;c++)
						weights[ofm][ifm][r][c] = (drand48()*2-1)/10;

		bias = new float [noutput_feature_map];
		for(int ofm=0; ofm<noutput_feature_map; ofm++)
			bias[ofm] = (drand48()*2-1)/10;

	}

	void backward(int batch_core, int starting_ind){	
		blasint KSIZE = nrow_conv;
		blasint DSIZE = nrow_output+2*(KSIZE-1);
		blasint I = noutput_feature_map/group;
		blasint O = ninput_feature_map/group;
		blasint MINIBATCHSIZE = mini_batch_size;
		blasint NCONV=(DSIZE-KSIZE)/stride+1;
		 
		blasint NDATAROW = I*KSIZE*KSIZE;
		blasint NDATACOL = NCONV * NCONV * MINIBATCHSIZE;
		blasint NKERNELROW = O;
		blasint OFFSET_KERN = NKERNELROW * NDATAROW;

		float alpha[] = {1.0, 1.0};
		float beta []= {0.0, 0.0};

		char no_trans = 'N';
		char trans = 'T';

		float * buf_grads;
		float * buf_grad_temp;
		buf_grads = (float *)malloc(sizeof(float) * O * NDATACOL);
		buf_grad_temp = (float *)malloc(sizeof(float) * NDATAROW * NDATACOL); 
		float * weight_diff = (float *)malloc(sizeof(float) * NDATAROW * O);
		float * buf_weight_temp = (float *)malloc(sizeof(float) * I * O * KSIZE*KSIZE);
		int indw=0;
		for(int ofm=0; ofm<I; ofm++)
			for(int ifm=0; ifm<O; ifm++)
				for(int r=KSIZE-1; r>=0; r--)
					for(int c=KSIZE-1; c>=0; c--)
						buf_weight_temp[indw++]=weights[ofm][ifm][r][c];

		for(int g=0;g<group;g++){
			int indx=0;
			for(int mb=0; mb<mini_batch_size; mb++)
				for(int fm=g*I; fm<(g+1)*I; fm++)
					for(int r=0;r<KSIZE;r++)
						for(int c=0; c<KSIZE; c++)
							for(int i=0; i<NCONV*stride; i+=stride)
								for(int j=0; j<NCONV*stride; j+=stride){
									if(i+r<KSIZE-1||i+r>=DSIZE-KSIZE+1||j+c<KSIZE-1||j+c>=DSIZE-KSIZE+1)
										buf_grad_temp[indx++] =0;
									else			
										buf_grad_temp[indx++] = grad[mb][fm][i+r-KSIZE+1][j+c-KSIZE+1];
								}


			BLASF(sgemm) (&no_trans, &no_trans, &NDATACOL, &O, &NDATAROW, alpha, buf_grad_temp, &NDATACOL, buf_weight_temp+g*OFFSET_KERN, &NDATAROW, beta, buf_grads, &NDATACOL);

			int idx =0;        
			for(int mb=0; mb<MINIBATCHSIZE; mb++)
				for(int fm=g*O; fm<(g+1)*O; fm++)
					for(int r=0; r<NCONV; r++)
						for(int c=0; c<NCONV; c++)
							grads[mb][fm][r][c] = buf_grads[idx++];
		}
			
		KSIZE = nrow_output;
		DSIZE = nrow_input+2*pad;
		I = ninput_feature_map/group;
		O = noutput_feature_map/group;
		MINIBATCHSIZE = mini_batch_size;
		NCONV=nrow_conv;

		NDATAROW = KSIZE*KSIZE*MINIBATCHSIZE;
		NDATACOL = NCONV * NCONV * I;// * MINIBATCHSIZE;
		NKERNELROW = O;
		OFFSET_KERN = NKERNELROW * NDATAROW;

		float * data;

		if (( data = (float *)malloc(sizeof(float) * NDATAROW * NDATACOL)) == NULL){
				fprintf(stderr,"Out of Memory!!\n");exit(1);
		}
		for(int g=0;g<group;g++){
			int indx=0;
			for(int mb=0; mb<MINIBATCHSIZE; mb++)
				for(int r=0; r<KSIZE; r++)
					for(int c=0; c<KSIZE; c++)
						for(int fm=g*I; fm<(g+1)*I; fm++)
							for(int i=0; i<NCONV*stride; i+=stride)
								for(int j=0; j<NCONV*stride; j+=stride){
									if(i+r<pad || i+r>=nrow_input+pad || j+c<pad || j+c>=ncol_input+pad)
										data[indx++]=0;
									else
										data[indx++]=inputs[mb][fm][i+r-pad][j+c-pad];
								}
			BLASF(sgemm) (&no_trans, &no_trans, &NDATACOL, &O, &NDATAROW, alpha, data, &NDATACOL, buf_grad, &NDATAROW, beta, weight_diff, &NDATACOL);

			for(int iter=g*O*NDATACOL;iter<(g+1)*O*NDATACOL;iter++){
				buf_weight[iter] = buf_weight[iter] - STEPSIZE* weight_diff[iter-g*O*NDATACOL];
			}                  								
		}	
	}

	void forward(int batch_core, int starting_ind){
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
			for(int f=0; f<I; f++)
				for(int r=0; r<KSIZE; r++)
					for(int s=0; s<KSIZE; s++)
						for(int i=0; i<DSIZE-KSIZE+1; i++)
							for(int j=0; j<DSIZE-KSIZE+1; j++){
								data[indx++]=inputs[mb][f][i+r][j+s];
								// cout << data[indx-1] << " ";
							}

		BLASF(sgemm) (&trans, &trans, &NDATACOL, &O, &NDATAROW, alpha, data, &NDATACOL, buf_weight, &NDATAROW, beta, buf_out, &NDATACOL );

		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
			for(int fm=0; fm<noutput_feature_map; fm++)
				for(int r=0;r<nrow_output;r++)
					for(int c=0; c<ncol_output; c++)
						output[mb][fm][r][c]+=bias[fm];
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
				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input, int _stride=1, int _pad=0, int _group=1):
		Operation(_mini_batch_size, _ninput_feature_map, _noutput_feature_map,
				_nrow_output,_ncol_output,_nrow_input,_ncol_input,_stride,_pad,_group){

		// assert(nrow_input % nrow_output == 0);
		// assert(ncol_input % ncol_output == 0);
			// TODO: NEED WORK

	}

	void backward(int batch_core, int starting_ind){
			// TODO: FLOAT (== check)
		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
			for(int ofm=0; ofm<noutput_feature_map; ofm++)
				for(int r=0;r<nrow_output;r++){
					for(int c=0;c<ncol_output;c++){
						float cvalue = output[mb][ofm][r][c];
						float cgrad = grad[mb][ofm][r][c];
						bool flag=0;
						for(int ir=r*stride;ir<r*stride+nrow_conv;ir++){
							for(int ic=c*stride;ic<c*stride+ncol_conv;ic++){
								if(ir>=pad && ir<nrow_input+pad && ic>=pad && ic<ncol_input+pad){
									if(inputs[mb][ofm][ir-pad][ic-pad] == cvalue && flag==0){
										grads[mb][ofm][ir-pad][ic-pad] += cgrad;
										flag=1;
									}else{
										grads[mb][ofm][ir-pad][ic-pad] = 0;
									}
								}
							}
						}
					}
				}

	}

	void forward(int batch_core, int starting_ind){
		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
			for(int ofm=0; ofm<noutput_feature_map; ofm++){
				for(int r=0;r<nrow_output;r++){
					for(int c=0;c<ncol_output;c++){
						float max = -10000;
						for(int ir=r*stride;ir<r*stride+nrow_conv;ir++){
							for(int ic=c*stride;ic<c*stride+ncol_conv;ic++){
								if(ir>=pad && ir<nrow_input+pad && ic>=pad && ic<ncol_input+pad){
									if(inputs[mb][ofm][ir][ic] > max){
										max = inputs[mb][ofm][ir-pad][ic-pad];
									}
								}
								else if(0 > max){
										max = 0;
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

	void backward(int batch_core, int starting_ind){
		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
			for(int label=0;label<n_label;label++){
				float cvalue = output[mb][label][0][0];
				for(int i_input=0;i_input<n_input;i_input++){

					float w = softweights[mb][label][i_input];
					float x = inputs[mb][i_input][0][0];

					float grad_w = (label == groundtruth[mb])*x - cvalue*x;
					float grad_x = (label == groundtruth[mb])*w - cvalue*w;
					// show(groundtruth[mb])

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

	void forward(int batch_core, int starting_ind){
		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
			for(int i=0;i<n_label;i++){
				float sum = 0.0;
				for(int i_input=0;i_input<n_input;i_input++){
					sum += softweights[mb][i][i_input] * inputs[mb][i_input][0][0];
				}
				sum += biases[i];
				output[mb][i][0][0] = sum;
			}

		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++){
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

	void backward(int batch_core, int starting_ind){
		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
			for(int ofm=0; ofm<noutput_feature_map; ofm++)
				for(int r=0;r<nrow_output;r++)
					for(int c=0;c<ncol_output;c++)
						if(output[mb][ofm][r][c]>0)
							grads[mb][ofm][r][c]=grad[mb][ofm][r][c];
						else
							grads[mb][ofm][r][c]=0;
	}

	void forward(int batch_core, int starting_ind){
		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
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

	void backward(int batch_core, int starting_ind){
		;
	}

	void forward(int batch_core, int starting_ind){
		;
	}


};

class DropoutOperation : public Operation{
public:
	float ratio;
	float scale;

	void clear_grad(){
		if(grads[0] != NULL)
			for(int mb=0; mb<mini_batch_size; mb++)
				for(int fm=0; fm<ninput_feature_map; fm++)
					for(int r=0;r<nrow_input;r++)
						for(int c=0; c<ncol_input;c++)
							grads[mb][fm][r][c] = 0;
	}

	DropoutOperation(int _mini_batch_size, int _ninput_feature_map, int _noutput_feature_map, 
				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input,float _ratio):
		Operation(_mini_batch_size, _ninput_feature_map, _noutput_feature_map,
				_nrow_output,_ncol_output,_nrow_input,_ncol_input){
			ratio = _ratio;
			scale = 1./(1.-ratio);

	}

	void backward(int batch_core, int starting_ind){
		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
			for(int ofm=0; ofm<noutput_feature_map; ofm++)
				for(int r=0;r<nrow_output;r++)
					for(int c=0;c<ncol_output;c++)
						if(output[mb][ofm][r][c]!=0)
							grads[mb][ofm][r][c]=grad[mb][ofm][r][c]*scale;
						else
							grads[mb][ofm][r][c]=0;
	}

	void forward(int batch_core, int starting_ind){
		default_random_engine generator;
		bernoulli_distribution distribution(ratio);
		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
			for(int fm=0; fm<noutput_feature_map; fm++)
				for(int r=0;r<nrow_output;r++)
					for(int c=0;c<ncol_output;c++){
						if (distribution(generator))
							output[mb][fm][r][c] = inputs[mb][fm][r][c]*scale;
						else
							output[mb][fm][r][c] = 0;
					}
	}
};

class LRNOperation : public Operation{
public:
	int local_size;
	float alpha;
	float beta;
	bool is_across;

	void clear_grad(){
		if(grads[0] != NULL)
			for(int mb=0; mb<mini_batch_size; mb++)
				for(int fm=0; fm<ninput_feature_map; fm++)
					for(int r=0;r<nrow_input;r++)
						for(int c=0; c<ncol_input;c++)
							grads[mb][fm][r][c] = 0;
	}

	LRNOperation(int _mini_batch_size, int _ninput_feature_map, int _noutput_feature_map, 
				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input, 
				int _local_size=5, float _alpha=0.0001, float _beta=0.75, bool _is_across=true):
		Operation(_mini_batch_size, _ninput_feature_map, _noutput_feature_map,
				_nrow_output,_ncol_output,_nrow_input,_ncol_input){
			local_size=_local_size;
			alpha=_alpha;
			beta=_beta;
			is_across=_is_across;
	}

	void backward(int batch_core, int starting_ind){
		if(is_across){
			for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
				for(int ofm=0; ofm<noutput_feature_map; ofm++)
					for(int r=0;r<nrow_output;r++)
						for(int c=0;c<ncol_output;c++){
							float cvalue = output[mb][ofm][r][c];
							float cgrad = grad[mb][ofm][r][c];

							int begin=max(0,ofm-local_size/2);
							int end=min(noutput_feature_map,ofm+local_size/2);
							for(int ifm=begin; ifm<end; ifm++)
								grads[mb][ifm][r][c]+=2*alpha*beta*inputs[mb][ofm][r][c]/local_size*pow(pow(cvalue,beta-1),1.0/beta)*cgrad;
						}
		}
		else{
			for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
				for(int ofm=0; ofm<noutput_feature_map; ofm++)
					for(int r=0;r<nrow_output;r++)
						for(int c=0;c<ncol_output;c++){
							float cvalue = output[mb][ofm][r][c];
							float cgrad = grad[mb][ofm][r][c];

							int i_begin=max(0,r-local_size/2);
							int i_end=min(noutput_feature_map,r+local_size/2);
							int j_begin=max(0,c-local_size/2);
							int j_end=min(noutput_feature_map,c+local_size/2);
							for(int i=i_begin; i<i_end; i++)
								for(int j=j_begin; j<j_end; j++)
									grads[mb][ofm][i][j]+=2*alpha*beta*inputs[mb][ofm][r][c]/local_size*pow(pow(cvalue,beta-1),1.0/beta)*cgrad;
						}
		}
	}

	void forward(int batch_core, int starting_ind){
		if(is_across){
			for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
				for(int ofm=0; ofm<noutput_feature_map; ofm++)
					for(int r=0;r<nrow_output;r++)
						for(int c=0;c<ncol_output;c++){
							int begin=max(0,ofm-local_size/2);
							int end=min(noutput_feature_map,ofm+local_size/2);
							float sum=0;
							for(int ifm=begin; ifm<end; ifm++)
								sum+=inputs[mb][ifm][r][c]*inputs[mb][ifm][r][c];
							sum=sum*alpha/local_size+1;
							output[mb][ofm][r][c]=inputs[mb][ofm][r][c]/pow(sum,beta);
						}
		}
		else{
			for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
				for(int ofm=0; ofm<noutput_feature_map; ofm++)
					for(int r=0;r<nrow_output;r++)
						for(int c=0;c<ncol_output;c++){
							int i_begin=max(0,r-local_size/2);
							int i_end=min(noutput_feature_map,r+local_size/2);
							int j_begin=max(0,c-local_size/2);
							int j_end=min(noutput_feature_map,c+local_size/2);
							float sum=0;
							for(int i=i_begin; i<i_end; i++)
								for(int j=j_begin; j<j_end; j++)
									sum+=inputs[mb][ofm][i][j]*inputs[mb][ofm][i][j];
							sum=sum*alpha/local_size+1;
							output[mb][ofm][r][c]=inputs[mb][ofm][r][c]/pow(sum,beta);
						}
		}
	}
};



#endif










