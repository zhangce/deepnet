#include "Image.h"
#include "Operation.h"

#ifndef _NETWORK_H
#define _NETWORK_H

class Layer{
public:

	Operation * operations;

	void forward(){
		Operation * const operation = operations;
		operation->forward();
	}

	void backward(){
		Operation * operation = operations;
		operation->backward();
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

	Network(int _n_layer){
		n_layer = _n_layer;
		layers = new Layer*[n_layer];
	}

	void forward(){
		for(int i_layer=0; i_layer<n_layer; i_layer++){
			Layer * const layer = layers[i_layer];
			layer->forward();
		}
	}

	void backward(){
		for(int i_layer=0; i_layer<n_layer; i_layer++){
			Layer * layer = layers[i_layer];
			layer->clear_grad();
		}

		for(int i_layer=n_layer-1;i_layer>=0;i_layer--){
			Layer * layer = layers[i_layer];
			layer->backward();
		}
	}
};


#endif

