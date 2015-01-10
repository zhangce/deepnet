#include "Image.h"
#include "Operation.h"

#ifndef _NETWORK_H
#define _NETWORK_H

class Layer{
public:

	Operation * operations;

	void forward(int max_core, int num_core_per_chunk){
		Operation * const operation = operations;
		operation->start_forward_thread(num_core_per_chunk, max_core);
	}

	void backward(int max_core,int num_core_per_chunk){
		Operation * operation = operations;
		operation->start_backward_thread(num_core_per_chunk, max_core);
	}

	void clear_grad(){
		Operation * operation = operations;
		operation->clear_grad();
	}

};


class Network{
public:

	int n_layer;
	Layer ** layers;
	int num_core_per_chunk;
	int max_core;

	Network(int _n_layer){
		n_layer = _n_layer;
		layers = new Layer*[n_layer];
		num_core_per_chunk=1;
		max_core=1;
	}

	void forward(){
		for(int i_layer=0; i_layer<n_layer; i_layer++){
			// show(i_layer);
			Layer * const layer = layers[i_layer];
			layer->forward(max_core,num_core_per_chunk);

		}
	}

	void backward(){
		for(int i_layer=0; i_layer<n_layer; i_layer++){
			Layer * layer = layers[i_layer];
			layer->clear_grad();
		}

		for(int i_layer=n_layer-1;i_layer>=0;i_layer--){
			// show(i_layer);
			Layer * layer = layers[i_layer];
			layer->backward(max_core,num_core_per_chunk);
		}
	}
};


#endif

