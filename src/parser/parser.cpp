
#include <leveldb/db.h>
#include "lmdb.h"
#include <stdint.h>
#include <iostream>
#include <glog/logging.h>
#include <fcntl.h>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message_lite.h>
#include "cnn.pb.h"

using namespace std;

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  google::protobuf::io::FileInputStream fileInput(fd);
  fileInput.SetCloseOnDelete( true );
  bool success = google::protobuf::TextFormat::Parse(&fileInput, proto);
  return success;
}


void ReadNetParamsFromTextFile(const string& param_file, Message* param) {
  ReadProtoFromTextFile(param_file.c_str(), param);
}


void dataSetup(cnn::LayerParameter& layer_param, cnn::Datum& datum){
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
 
  switch (layer_param.data_param().backend()) {
  case 1:
    mdb_env_create(&mdb_env_);
    mdb_env_set_mapsize(mdb_env_, 1099511627776);
    mdb_env_open(mdb_env_, layer_param.data_param().source().c_str(), MDB_RDONLY|MDB_NOTLS, 0664);
    mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_);
    mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_);
    mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_);
    mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST);
    break;
  default:
    break;
  }

  // Read a data point, and use it to initialize the top blob.
  switch (layer_param.data_param().backend()) {
  case 1:
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    break;
  }
}

int main(){
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	cnn::SolverParameter solver_param;
	ReadProtoFromTextFile("mnist/lenet_solver.prototxt", &solver_param);
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
    } 
    
    if(layer_param.type() ==4){
      std::cout << layer_param.convolution_param().kernel_size() << std::endl;
      std::cout << layer_param.convolution_param().num_output() << std::endl;
      std::cout << layer_param.convolution_param().pad() << std::endl;
      std::cout << layer_param.convolution_param().stride() << std::endl;
    }
    
  }
}
