#ifndef _CORPUS_H
#define _CORPUS_H

#include <string>
#include <stdio.h>

#include "Image.h"

#include <stdint.h>
#include <arpa/inet.h>
#include <fstream>

class MNISTCorpus{
public:

	int n_image;
	int n_batch;
	int n_rows;
	int n_cols;
	Image ** images;

	MNISTCorpus(std::string label_file, std::string img_file, int mini_batch_size, int dim){

		int padding = 2;

	    std::ifstream file (img_file.c_str(), std::ios::binary);
	    std::ifstream labelfile (label_file.c_str(), std::ios::binary);

	    int magic_number=0;
	    int label = 0;
	    unsigned char cc = ' ';
	    unsigned char temp = ' ';

		file.read((char*)&magic_number,sizeof(magic_number));
		file.read((char*)&n_image,sizeof(n_image));
		file.read((char*)&n_rows,sizeof(n_rows));
	    file.read((char*)&n_cols,sizeof(n_cols));

	    labelfile.read((char*)&magic_number,sizeof(magic_number));
	    labelfile.read((char*)&n_image,sizeof(n_image));

	    magic_number= ntohl(magic_number);
	    n_image= ntohl(n_image);
	    n_rows= ntohl(n_rows) + padding*2;
	    n_cols= ntohl(n_cols) + padding*2;

	    images = new Image*[n_image];

	    n_batch = n_image/mini_batch_size;

		for(int i=0;i<n_batch;i++){
	    	labelfile.read((char*)&cc,sizeof(cc));
	    	label = unsigned(cc);

	    	images[i] = new Image(mini_batch_size ,dim,i, label, n_rows, n_cols);

	    	for(int b=0;b<mini_batch_size;++b){
	    		for(int d=0;d<dim;++d){
			    	for(int r=0;r<n_rows-padding*2;++r){
			    		for(int c=0;c<n_cols-padding*2;++c){
			    			file.read((char*)&temp,sizeof(temp));
			    			//images[i]->pixels[r][c] = 1.0*unsigned(temp)/255;
			    			images[i]->pixels[b][d][r+padding][c+padding] = 1.0*unsigned(temp)/255;
			    		}
			    	}
			    }
			} 	
		}
	}
};

#endif

