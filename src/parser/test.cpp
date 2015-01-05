#include <iostream>
#include <fstream>
#include <string>
#include "cnn.pb.h"
#include "parser.hpp"

using namespace std;

int main(int argc, char ** argv){

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
}
