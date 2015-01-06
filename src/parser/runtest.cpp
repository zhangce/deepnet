#include <iostream>
#include <fstream>
#include <string>
#include "cnn.pb.h"
#include "parser.hpp"

using namespace std;

int main(int argc, char ** argv){

  if(argc < 2){
    cout << "ERROR: No input file provided" << endl;
    return 0;
  }

  cout << "Reading file with Solver Parameters" << endl;
	cnn::SolverParameter solver_param;
  cnn::NetParameter net_param;
  cnn::Datum data;
	ReadProtoFromTextFile(argv[1], &solver_param);
  cout << "Reading file with Network Definition" << endl;
  if(!solver_param.has_net()){
    cout << "ERROR: No Network file provided" << endl;
    return 0;
  }
  cout << "Network Definition File: " << solver_param.net() << endl;
  ReadNetParamsFromTextFile(solver_param.net(), &net_param);  
  cout << "Reading the network parameters done" << endl;

  if(net_param.has_name()){
    cout << "Network Name: " << net_param.name() << endl;
  }

if(net_param.layers_size() == 0){
    cout << "Network has no layers" << endl;
    return 0;
} 
 
cout << "Total layers in the network: " << net_param.layers_size() << endl;

for (int i = 0; i < net_param.layers_size(); ++i) {
  cnn::LayerParameter layer_param = net_param.layers(i); 
  //const string& type = cnn::LayerParameter::LayerType_Name(layer_param.type());
  int type = layer_param.type();
  switch (type){
  case cnn::LayerParameter_LayerType_CONVOLUTION:  
    cout << "Testing Convolutional Layer" << endl;
    if (layer_param.top_size()== 0 || layer_param.bottom_size() == 0){
      cout << "Layer should have both input and output parameter defined" << endl;
      return 0;
    }
    break;
  case cnn::LayerParameter_LayerType_POOLING:
    cout << "Testing Pooling Layer" << endl;
    if (layer_param.top_size()== 0 || layer_param.bottom_size() == 0){
      cout << "Layer should have both input and output parameter defined" << endl;
      return 0;
    }
    break;
  case cnn::LayerParameter_LayerType_RELU:
    cout << "Testing RELU Layer" << endl;
    if (layer_param.top_size()== 0 || layer_param.bottom_size() == 0){
      cout << "Layer should have both input and output parameter defined" << endl;
      return 0;
    }
    break;
  case cnn::LayerParameter_LayerType_INNER_PRODUCT:
    cout << "Testing Inner Product Layer" << endl;
    if (layer_param.top_size()== 0 || layer_param.bottom_size() == 0){
      cout << "Layer should have both input and output parameter defined" << endl;
      return 0;
    }
    break;
  case cnn::LayerParameter_LayerType_DATA:{
    cout << "Testing Data Layer" << endl;
    if(!layer_param.data_param().has_source()){
      cout << "ERROR: No data source provided" << endl;
      return 0;
    }
    ifstream ifile(layer_param.data_param().source().c_str());
    if(!ifile){
      cout << "ERROR: Data source could not be found" << endl;
      return 0;
    }
    dataSetup(layer_param, data);
    //cout << cnn::Phase_Name(layer_param.include(0).phase()) << endl;
    break;
   }
  case cnn::LayerParameter_LayerType_LRN:
    cout << "Testing LRN Layer" << endl;
    if (layer_param.top_size()== 0 || layer_param.bottom_size() == 0){
      cout << "Layer should have both input and output parameter defined" << endl;
      return 0;
    }
    break;
  case cnn::LayerParameter_LayerType_SOFTMAX_LOSS:
    cout << "Testing loss Layer" << endl;
    if (layer_param.top_size()== 0 || layer_param.bottom_size() == 0){
      cout << "Layer should have both input and output parameter defined" << endl;
      return 0;
    }
    break; 
  case cnn::LayerParameter_LayerType_DROPOUT:
    cout << "Testing Dropout Layer" << endl;
    if (layer_param.top_size()== 0 || layer_param.bottom_size() == 0){
      cout << "Layer should have both input and output parameter defined" << endl;
      return 0;
    }
    break;
  case cnn::LayerParameter_LayerType_ACCURACY:
    cout << "Testing Accuracy Layer" << endl;
    if (layer_param.top_size()== 0 || layer_param.bottom_size() == 0){
      cout << "Layer should have both input and output parameter defined" << endl;
      return 0;
    }
    break;
  default:
    cout << "Invalid Layer type" << endl;
    break;        
  }
  //std::cout << layer_param.top_size() << std::endl;
  //std::cout << layer_param.bottom_size() << std::endl;
  /*
  for(int top_id = 0; top_id < layer_param.top_size() ; ++top_id){
    //std::cout << layer_param.top(top_id) << std::endl;
  }
  std::cout << "\n";
  std::cout << cnn::LayerParameter::LayerType_Name(layer_param.type()) << std::endl;

  cnn::Datum data;
  if(type == 5){
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
  */
}
}
