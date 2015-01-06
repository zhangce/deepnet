
#include <iostream>

#ifndef _IMAGE_H
#define _IMAGE_H

class Image{
public:
	int bs;
	int dim;
	int img_id;
	int * label;
	int nrows;
	int ncols;

	float **** pixels;

	float * _buf;

	Image(int _bs, int _dim, int _img_id, int * _label, int _nrow, int _ncol){
		bs = _bs;
		dim = _dim;
		img_id = _img_id;
		nrows = _nrow;
		ncols = _ncol;
		label = _label;
		_buf = new float[bs*dim*nrows*ncols];
		pixels = new float ***[bs];
		for(int b=0; b<bs;b++){
			pixels[b] = new float ** [dim];
			for(int d=0; d<dim; d++){
				pixels[b][d] = new float*[nrows];
				for(int i=0;i<nrows;i++){
					pixels[b][d][i] = &_buf[b*dim*nrows*ncols+d*nrows*ncols+i*ncols];
				}
			}
		}
	}

	void show(){
		for(int b=0;b<bs;b++){
			for(int d=0; d<dim; d++){
				for(int r=0;r<nrows;r++){
					for(int c=0;c<ncols;c++){
						std::cout << pixels[b][d][r][c] <<  " " ;
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
	}

};

#endif