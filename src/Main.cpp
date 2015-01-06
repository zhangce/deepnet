
#include "Image.h"
#include <iostream>

#include "Corpus.h"
#include "Operation.h"
#include "Network.h"

#include "timer.h"
#include "parser/parser.h"
using namespace std;

int DIGIT=10;

void LeNet5();


int main(int argc, char ** argv){

	MNISTCorpus corpus("input/train-labels-idx1-ubyte", "input/train-images-idx3-ubyte");
	cnn::SolverParameter solver_param;
	ReadProtoFromTextFile(argv[1], &solver_param);
	cnn::NetParameter net_param;
	ReadNetParamsFromTextFile(solver_param.net(), &net_param);
	
	for (int i = 0; i < net_param.layers_size(); ++i) {
	    //std::cout << net_param.layers(i).type() << std::endl;
	    cnn::LayerParameter layer_param = net_param.layers(i); 
	    //std::cout << layer_param.top_size() << std::endl;
	    //std::cout << layer_param.bottom_size() << std::endl;
	    
	    for(int top_id = 0; top_id < layer_param.top_size() ; ++top_id){
	      //std::cout << layer_param.top(top_id) << std::endl;
	    }
	    std::cout << "\n";
	    std::cout << cnn::LayerParameter::LayerType_Name(layer_param.type()) << std::endl;
	    const string& type = cnn::LayerParameter::LayerType_Name(layer_param.type());

	    cnn::Datum data;
	    if(type == "DATA"){
	    	//std::cout << layer_param.data_param().backend() << std::endl;
	      std::cout << layer_param.data_param().batch_size() << std::endl;
	    	dataSetup(layer_param, data);
	      std::cout << data.channels() << std::endl;
	      std::cout << data.height() << std::endl;
	      std::cout << data.width() << std::endl;
	      std::cout << data.label() << std::endl;
	    } 
	    
	    if(layer_param.type() ==4){
	      std::cout << layer_param.convolution_param().kernel_size() << std::endl;
	      std::cout << layer_param.convolution_param().num_output() << std::endl;
	      std::cout << layer_param.convolution_param().pad() << std::endl;
	      std::cout << layer_param.convolution_param().stride() << std::endl;
	    }
	    
	  }
	LeNet5();
	return 0;

	// Build Network

	// Network network(1);
	// network.layers[0] = new Layer(6);
	// Layer * layer1 = network.layers[0];
	// for(int i=0;i<6;i++){
	// 	layer1->operations[i]
	// 		= (Operation*) new FullyConnectedOperation(1, 24, 24, corpus.n_rows, corpus.n_cols);
	// 	layer1->operations[i]->inputs[0] = corpus.images[0]->pixels;
	// 	layer1->operations[i]->grads[0] = NULL;
	// }

	// for(int i_epoch=0;i_epoch<1;i_epoch++){
	// 	cerr << i_epoch << endl;
	// 	int ncorr[10];
	// 	int ncorr_neg[10];
	// 	int npos[10];
	// 	int nneg[10];
	// 	for(int i=0;i<10;i++){
	// 		ncorr[i] = 0;
	// 		ncorr_neg[i] = 0;
	// 		npos[i] = 0;
	// 		nneg[i] = 0;
	// 	}
	// 	float loss = 0.0;
	// 	float loss_test = 0.0;

	// 	Timer t;
	// 	for(int i_img=0;i_img<corpus.n_image;i_img++){
	// 		for(int i=0;i<4;i++){
	// 			layer1->operations[i]->inputs[0] 
	// 				= corpus.images[i_img]->pixels;
	// 		}
	// 		network.forward();
	// 	}
	// 	float trainingtime = t.elapsed();
	// 	std::cout << "Training " << trainingtime << " seconds..." << "  " <<
	// 		(trainingtime/corpus.n_image) << " seconds/image." << std::endl;
	// 	float throughput = 1.0*corpus.n_rows*corpus.n_cols*sizeof(float)*corpus.n_image/1024/1024/trainingtime;
	// 	std::cout << "     THROUGHPUT = " << throughput << "MB/seconds..." << std::endl;
	// }

	// std::cout << "DONE" << std::endl;

	return 0;
}


void LeNet5(){

	MNISTCorpus corpus("input/train-labels-idx1-ubyte", "input/train-images-idx3-ubyte");
	MNISTCorpus corpus_test("input/t10k-labels-idx1-ubyte", "input/t10k-images-idx3-ubyte");

	// Build Network

	Network network(6);

	network.layers[0] = new Layer();
	Layer * layer1 = network.layers[0];
	layer1->operations
		= (Operation*) new ConvOperation(1,1,20,24,24,28,28);
	layer1->operations->inputs = corpus.images[0]->pixels;
	layer1->operations->grads = NULL;


	// network.layers[1] = new Layer();
	// Layer * layer2 = network.layers[1];
	// layer2->operations
	// 	= (Operation*) new MaxPoolingOperation(1, 12, 12, 24, 24);
	// layer2->operations->inputs = layer1->operations->output;
	// layer2->operations->grads = layer1->operations->grad;



	// network.layers[2] = new Layer(32);
	// Layer * layer3 = network.layers[2];
	// for(int i=0;i<32;i++){
	// 	layer3->operations[i]
	// 		= (Operation*) new FullyConnectedOperation(6, 8, 8, 12, 12);
	// 	for(int j=0;j<6;j++){
	// 		layer3->operations[i]->inputs[j] = layer2->operations[j]->output;
	// 		layer3->operations[i]->grads[j] = layer2->operations[j]->grad;
	// 	}
	// }

	// network.layers[3] = new Layer(32);
	// Layer * layer4 = network.layers[3];
	// for(int i=0;i<32;i++){
	// 	layer4->operations[i]
	// 		= (Operation*) new MaxPoolingOperation(1, 4, 4, 8, 8);
	// 	layer4->operations[i]->inputs[0] = layer3->operations[i]->output;
	// 	layer4->operations[i]->grads[0] = layer3->operations[i]->grad;
	// }

	// int NLAYER5=120;
	// network.layers[4] = new Layer(NLAYER5);
	// Layer * layer5 = network.layers[4];
	// for(int i=0;i<NLAYER5;i++){
	// 	layer5->operations[i]
	// 		= (Operation*) new FullyConnectedOperation(32, 1, 1, 4, 4);
	// 	for(int j=0;j<32;j++){
	// 		layer5->operations[i]->inputs[j] = layer4->operations[j]->output;
	// 		layer5->operations[i]->grads[j] = layer4->operations[j]->grad;
	// 	}
	// }

	// network.layers[5] = new Layer(1);
	// Layer * layer6 = network.layers[5];
	// layer6->operations[0]
	// 	= (Operation*) new SoftmaxOperation(NLAYER5, DIGIT, 1, 1);
	// for(int i=0;i<NLAYER5;i++){
	// 	layer6->operations[0]->inputs[i]
	// 		= layer5->operations[i]->output;
	// 	layer6->operations[0]->grads[i]
	// 		= layer5->operations[i]->grad;
	// }

	// for(int i_epoch=0;i_epoch<1;i_epoch++){
	// 	int ncorr[10];
	// 	int ncorr_neg[10];
	// 	int npos[10];
	// 	int nneg[10];
	// 	for(int i=0;i<10;i++){
	// 		ncorr[i] = 0;
	// 		ncorr_neg[i] = 0;
	// 		npos[i] = 0;
	// 		nneg[i] = 0;
	// 	}
	// 	float loss = 0.0;
	// 	float loss_test = 0.0;

	// 	Timer t;
	// 	cout << corpus.n_image << endl;
	// 	for(int i_img=0;i_img<corpus.n_image;i_img++){

	// 		layer6->operations[0]->groundtruth 
	// 			= (corpus.images[i_img]->label);
	// 		for(int i=0;i<4;i++){
	// 			layer1->operations[i]->inputs[0] 
	// 				= corpus.images[i_img]->pixels;
	// 		}

	// 		network.forward();
	// 		//network.backward();	
	// 	}
	// 	float trainingtime = t.elapsed();
	// 	std::cout << "Training " << trainingtime << " seconds..." << "  " <<
	// 		(trainingtime/corpus.n_image) << " seconds/image." << std::endl;

	// 	t.restart();
	// 	for(int i_img=0;i_img<corpus_test.n_image;i_img++){

	// 		layer6->operations[0]->groundtruth 
	// 			= (corpus_test.images[i_img]->label);
	// 		for(int i=0;i<4;i++){
	// 			layer1->operations[i]->inputs[0] 
	// 				= corpus_test.images[i_img]->pixels;
	// 		}

	// 		network.forward();
			
	// 		int gt = (corpus_test.images[i_img]->label);
	// 		int imax;
	// 		float ifloat = -1;
	// 		for(int dig=0;dig<DIGIT;dig++){
	// 			float out = layer6->operations[0]->output[0][dig];
	// 			if(out > ifloat){
	// 				imax = dig;
	// 				ifloat = out;
	// 			}
	// 		}
	// 		nneg[gt] ++;
	// 		ncorr_neg[gt] += (gt==imax);
	// 		loss_test += (gt==imax);
	// 	}
	// 	float testingtime = t.elapsed();
	// 	std::cout << "Testing " << t.elapsed() << " seconds..." << "  " <<
	// 		(testingtime/corpus_test.n_image) << " seconds/image." << std::endl;
		
	// 	std::cout << "----TEST----" << loss_test/corpus_test.n_image << std::endl;
	// 	for(int dig=0;dig<DIGIT;dig++){
	// 		std::cout << "## DIG=" << dig << " : ";
	// 		std::cout << 1.0*ncorr_neg[dig]/nneg[dig] << " = " << ncorr_neg[dig] << "/" << nneg[dig] << std::endl;
	// 	}

	// }

	std::cout << "DONE" << std::endl;

}



	/*
	Network network(3);

	network.layers[0] = new Layer(2);
	for(int i=0;i<2;i++){
		network.layers[0]->operations[i]
			= (Operation*) new ConvOperation(5, 5, corpus.n_rows, corpus.n_cols);
		network.layers[0]->operations[i]->inputs[0]
			= corpus.images[0]->pixels;
		network.layers[0]->operations[i]->grads[0]
			= NULL;	
	}

	network.layers[1] = new Layer(10);
	for(int i=0;i<10;i++){
		network.layers[1]->operations[i]
			= (Operation*) new FullyConnectedOperation(2, 1, 1, 5, 5);
		network.layers[1]->operations[i]->inputs[0]
			= network.layers[0]->operations[0]->output;
		network.layers[1]->operations[i]->inputs[1]
			= network.layers[0]->operations[1]->output;

		network.layers[1]->operations[i]->grads[0]
			= network.layers[0]->operations[0]->grad;
		network.layers[1]->operations[i]->grads[1]
			= network.layers[0]->operations[1]->grad;
	}

	network.layers[2] = new Layer(1);
	network.layers[2]->operations[0]
		= (Operation*) new LogisticOperation(10, 1, 1, 1, 1);
	for(int i=0;i<10;i++){
		network.layers[2]->operations[0]->inputs[i]
			= network.layers[1]->operations[i]->output;
		network.layers[2]->operations[0]->grads[i]
			= network.layers[1]->operations[i]->grad;
	}

	for(int i_epoch=0;i_epoch<100;i_epoch++){
		int ncorr = 0;
		int ncorr_neg = 0;
		int npos = 0;
		int nneg = 0;
		float loss = 0.0;
		for(int i_img=0;i_img<corpus.n_image;i_img++){
			network.layers[2]->operations[0]->groundtruth 
				= corpus.images[i_img]->label == 3;
			network.layers[0]->operations[0]->inputs[0]
				= corpus.images[i_img]->pixels;
			network.layers[0]->operations[1]->inputs[0]
				= corpus.images[i_img]->pixels;
			network.forward();

			int gt = network.layers[2]->operations[0]->groundtruth;
			float out = network.layers[2]->operations[0]->output[0][0];

			
			if(gt == 1){
				npos ++;
				ncorr += ((out>0.5) == gt);
			}else{
				nneg ++;
				ncorr_neg += ((out>0.5) == gt);
			}

			loss += fabs(gt-out);

			network.backward();
		}
		std::cout << "------------" << loss/corpus.n_image << std::endl;
		std::cout << "+ " << 1.0*ncorr/npos << " = " << ncorr << "/" << npos << std::endl;
		std::cout << "- " << 1.0*ncorr_neg/nneg << " = " << ncorr_neg << "/" << nneg << std::endl;
	}
	*/







