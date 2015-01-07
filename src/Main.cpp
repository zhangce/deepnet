
#include "Image.h"
#include <iostream>

#include "Corpus.h"
#include "Operation.h"
#include "Network.h"

#include "timer.h"
#include "parser/parser.h"
using namespace std;

int DIGIT=10;

void LeNet5(char * file);


int main(int argc, char ** argv){
	LeNet5(argv[1]);
	return 0;
}


void LeNet5(char * file){
	
	cnn::SolverParameter solver_param;
	ReadProtoFromTextFile(file, &solver_param);
	cnn::NetParameter net_param;
	ReadNetParamsFromTextFile(solver_param.net(), &net_param);
	
	// Build Network
	cnn::Datum train_data;
	cnn::Datum test_data;
	int n_label = 10;

	int mini_batch_size_train;
	int mini_batch_size_test;
	int ninput_feature_map, noutput_feature_map, nrow_output, ncol_output, nrow_input, ncol_input;
	int nrow_conv, ncol_conv;
	int pad, stride;
	int nlayers = net_param.layers_size();
		
	for (int i=0; i<nlayers; i++){
		cnn::LayerParameter layer_param = net_param.layers(i); 
		if(layer_param.type() == cnn::LayerParameter_LayerType_DATA){
			if (layer_param.include(0).phase() == 0){
				dataSetup(layer_param, train_data);
				mini_batch_size_train = layer_param.data_param().batch_size();
				ninput_feature_map = train_data.channels();
				nrow_input = train_data.height();
				ncol_input = train_data.width();
			}
			if (layer_param.include(0).phase() == 1){
				dataSetup(layer_param, test_data);
				mini_batch_size_test = layer_param.data_param().batch_size();
				mini_batch_size_test=mini_batch_size_train;
			}
			nlayers--;
		}
	}

	MNISTCorpus corpus("input/train-labels-idx1-ubyte", "input/train-images-idx3-ubyte", mini_batch_size_train, ninput_feature_map);
	cout << "Corpus train loaded" << endl;
	MNISTCorpus corpus_test("input/t10k-labels-idx1-ubyte", "input/t10k-images-idx3-ubyte", mini_batch_size_test, ninput_feature_map);
	cout << "Corpus test Loaded" << endl;	

	Network network(nlayers);

	cout << "Initialising data layer" << endl;
	Layer * dataLayer = new Layer();
	dataLayer->operations = (Operation*) new DataOperation(mini_batch_size_train, ninput_feature_map, ninput_feature_map, corpus.n_rows, corpus.n_cols, corpus.n_rows, corpus.n_cols);
	dataLayer->operations->output = corpus.images[0]->pixels;
	cout << "Batch 0 loaded" << endl;

	cout << "data initialised" << endl;
	
	for (int l=0; l<nlayers; l++){
		cout << "L" << l << endl;
		network.layers[l] = new Layer();
		Layer * layerp;
		if(l==0){
			layerp = dataLayer;	
		}
		else{
			layerp = network.layers[l-1];
		}
		Layer * layer = network.layers[l];
		
		cnn::LayerParameter layer_param = net_param.layers(l+2); 
		int type = layer_param.type();
		switch (type){
		case cnn::LayerParameter_LayerType_CONVOLUTION:  
		cout << "Convolution Operation" << endl;
		ncol_conv = layer_param.convolution_param().kernel_size();
		nrow_conv = layer_param.convolution_param().kernel_size();
		noutput_feature_map = layer_param.convolution_param().num_output();
		stride = layer_param.convolution_param().stride();
		nrow_output = (nrow_input - nrow_conv)/stride + 1;
		ncol_output = (ncol_input - ncol_conv)/stride + 1;

		layer->operations = (Operation*) new ConvOperation(mini_batch_size_train, ninput_feature_map, noutput_feature_map, nrow_output, ncol_output, nrow_input, ncol_input);
	        layer->operations->inputs = layerp->operations->output;
		layer->operations->grads = layerp->operations->grad;

		nrow_input = nrow_output;
		ncol_input = ncol_output;
		ninput_feature_map = noutput_feature_map;
		break;

		case cnn::LayerParameter_LayerType_POOLING:
		cout << "Max Pooling Operation" << endl;
		ncol_conv = layer_param.pooling_param().kernel_size();
		nrow_conv = layer_param.pooling_param().kernel_size();
		noutput_feature_map = ninput_feature_map;
		stride = layer_param.pooling_param().stride();
		nrow_output = (nrow_input - nrow_conv)/stride + 1;
		ncol_output = (ncol_input - ncol_conv)/stride + 1;

		layer->operations = (Operation*) new MaxPoolingOperation(mini_batch_size_train, ninput_feature_map, noutput_feature_map, nrow_output, ncol_output, nrow_input, ncol_input);
	    layer->operations->inputs = layerp->operations->output;
		layer->operations->grads = layerp->operations->grad;

		nrow_input = nrow_output;
		ncol_input = ncol_output;
		ninput_feature_map = noutput_feature_map;
		break;

		case cnn::LayerParameter_LayerType_RELU:
		cout << "Relu Operation" << endl;
		noutput_feature_map = ninput_feature_map;
		nrow_output = nrow_input;
		ncol_output = ncol_input;

		layer->operations = (Operation*) new RELUOperation(mini_batch_size_train, ninput_feature_map, noutput_feature_map, nrow_output, ncol_output, nrow_input, ncol_input);
	    layer->operations->inputs = layerp->operations->output;
		layer->operations->grads = layerp->operations->grad;

		nrow_input = nrow_output;
		ncol_input = ncol_output;
		ninput_feature_map = noutput_feature_map;
		break;

		case cnn::LayerParameter_LayerType_INNER_PRODUCT:
		cout << "Fully Connected Operation" << endl;
		noutput_feature_map = layer_param.inner_product_param().num_output();
		nrow_output = 1;
		ncol_output = 1;

		layer->operations = (Operation*) new FullyConnectedOperation(mini_batch_size_train, ninput_feature_map, noutput_feature_map, nrow_output, ncol_output, nrow_input, ncol_input);
	    layer->operations->inputs = layerp->operations->output;
		layer->operations->grads = layerp->operations->grad;

		nrow_input = nrow_output;
		ncol_input = ncol_output;
		ninput_feature_map = noutput_feature_map;
		break;

		case cnn::LayerParameter_LayerType_SOFTMAX_LOSS:
		cout << "Softmax Operation" << endl;
		noutput_feature_map = n_label;
		nrow_output = 1;
		ncol_output = 1;

		layer->operations = (Operation*) new SoftmaxOperation(mini_batch_size_train, ninput_feature_map, noutput_feature_map, nrow_output, ncol_output, nrow_input, ncol_input);
	    layer->operations->inputs = layerp->operations->output;
		layer->operations->grads = layerp->operations->grad;

		nrow_input = nrow_output;
		ncol_input = ncol_output;
		ninput_feature_map = noutput_feature_map;
		break; 
		
		case cnn::LayerParameter_LayerType_ACCURACY:
		break;

		default:
		cout << "Invalid Layer type" << endl;
		break;        
		}
	}

	for(int i_epoch=0;i_epoch<1;i_epoch++){
		int ncorr[10];
		int ncorr_neg[10];
		int npos[10];
		int nneg[10];
		for(int i=0;i<10;i++){
			ncorr[i] = 0;
			ncorr_neg[i] = 0;
			npos[i] = 0;
			nneg[i] = 0;
		}
		float loss = 0.0;
		float loss_test = 0.0;

		Timer t;
		cout << corpus.n_image << endl;
		for(int i_img=0;i_img<corpus.n_batch;i_img++){

			network.layers[nlayers-1]->operations->groundtruth = (corpus.images[i_img]->label);
			network.layers[0]->operations->inputs = corpus.images[i_img]->pixels;

			network.forward();
					float trainingtime = t.elapsed();
		std::cout << "Training " << trainingtime << " seconds..." << "  " <<
			(trainingtime/corpus.n_image) << " seconds/image." << std::endl;

			network.backward();
		}
		float trainingtime = t.elapsed();
		std::cout << "Training " << trainingtime << " seconds..." << "  " <<
			(trainingtime/corpus.n_image) << " seconds/image." << std::endl;

		t.restart();
		for(int i_img=0;i_img<corpus_test.n_batch;i_img++){
			network.layers[nlayers-1]->operations->groundtruth 
				= (corpus_test.images[i_img]->label);
			network.layers[0]->operations->inputs 
				= corpus_test.images[i_img]->pixels;
			show("BEF")
			network.forward();
		 	
			for(int img=0; img < mini_batch_size_test; img++){
				int gt = (corpus_test.images[i_img]->label[img]);
				int imax;
				float ifloat = -1;
				for(int dig=0;dig<DIGIT;dig++){
					float out = network.layers[nlayers-1]->operations->output[img][dig][0][0];
					if(out > ifloat){
						imax = dig;
						ifloat = out;
					}
				}
				nneg[gt] ++;
				ncorr_neg[gt] += (gt==imax);
				loss_test += (gt==imax);
			}	
		}
		float testingtime = t.elapsed();
		std::cout << "Testing " << t.elapsed() << " seconds..." << "  " <<
			(testingtime/corpus_test.n_image) << " seconds/image." << std::endl;
		
		std::cout << "----TEST----" << loss_test/corpus_test.n_image << std::endl;
		for(int dig=0;dig<DIGIT;dig++){
			std::cout << "## DIG=" << dig << " : ";
			std::cout << 1.0*ncorr_neg[dig]/nneg[dig] << " = " << ncorr_neg[dig] << "/" << nneg[dig] << std::endl;
		}
	}

	std::cout << "DONE" << std::endl;

}
