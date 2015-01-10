#ifndef _CORPUS_H
#define _CORPUS_H

#include <string>
#include <stdio.h>

#include "Image.h"

#include <stdint.h>
#include <arpa/inet.h>
#include <fstream>
#include "parser/parser.h"
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
#include "parser/cnn.pb.h"

using namespace std;

class Corpus{
public:

	int n_image;
	int n_batch;
	int n_rows;
	int n_cols;
	int dim;
	int mini_batch_size;
	Image ** images;

	Corpus(cnn::LayerParameter& layer_param){
		cnn::Datum datum;
		MDB_env* mdb_env_;
		MDB_dbi mdb_dbi_;
		MDB_txn* mdb_txn_;
		MDB_cursor* mdb_cursor_;
		MDB_val mdb_key_, mdb_value_;
		MDB_stat stat;
		
		switch (layer_param.data_param().backend()) {
		case 1:
		    mdb_env_create(&mdb_env_);
		    mdb_env_set_mapsize(mdb_env_, 1099511627776);
		    cout << layer_param.data_param().source().c_str() << endl;	
		    mdb_env_open(mdb_env_, layer_param.data_param().source().c_str(), MDB_RDONLY|MDB_NOTLS, 0664);
		    mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_);
		    mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_);
		    mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_);
		    mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST);
		    break;
		default:
		    break;
		}

		switch (layer_param.data_param().backend()) {
		case 1:
		    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
		    break;
		default:
		    break;
		}

		dim = datum.channels();
		n_rows = datum.height();
		n_cols = datum.width();
		mini_batch_size = layer_param.data_param().batch_size();
		int padding = 0;

		n_rows= n_rows + padding*2;
		n_cols= n_cols + padding*2;

		mdb_env_stat (mdb_env_, &stat);
  		n_image = stat.ms_entries;
  		n_image=2;

//		images = new Image*[n_image];

		n_batch = n_image/mini_batch_size;
		cout << n_batch << endl;
		images = new Image*[n_batch];

		MDB_cursor_op op = MDB_FIRST;
  		int row = n_rows-padding*2;
  		int col = n_cols-padding*2;

		for(int i=0;i<n_batch;i++){
			int * label = new int [mini_batch_size];
			images[i] = new Image(mini_batch_size, dim, i, label, n_rows, n_cols);
			for(int b=0;b<mini_batch_size;++b){
				mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
				datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
				const string& data = datum.data();
				op = MDB_NEXT;
				//labelfile.read((char*)&cc,sizeof(cc));
				int img_label = datum.label();
				label[b]=img_label;
				for(int d=0;d<dim;++d){
					for(int r=0;r<row;++r){
						for(int c=0;c<col;++c){
							//file.read((char*)&temp,sizeof(temp));
							float datum_element = static_cast<float>(static_cast<uint8_t>(data[d*row*col+r*col+c]));
							images[i]->pixels[b][d][r+padding][c+padding] = 1.0*unsigned(datum_element)/255;
						}
					}
				}
			}
		}
	}
};

#endif
