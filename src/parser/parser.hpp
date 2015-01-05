#include <string>

#include "cnn.pb.h"

using namespace std;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* solver_file, Message* proto);
void ReadNetParamsFromTextFile(const string& param_file, Message* param);
void dataSetup(cnn::LayerParameter& layer_param, cnn::Datum& datum);

