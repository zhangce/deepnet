
#include <iostream>

#ifndef _IMAGE_H
#define _IMAGE_H

class Image{
public:
	int dim;
	int img_id;
	int label;
	int nrows;
	int ncols;

	float *** pixels;

	float * _buf;

	Image(int _dim, int _img_id, int _label, int _nrow, int _ncol){
		dim = _dim;
		img_id = _img_id;
		nrows = _nrow;
		ncols = _ncol;
		label = _label;
		_buf = new float[1*dim*nrows*ncols];
		pixels = new float ** [dim];
		for(int d=0; d<dim; d++){
			pixels[d] = new float*[nrows];
			for(int i=0;i<nrows;i++){
				pixels[d][i] = &_buf[i*ncols];
			}
		}
	}

	void show(){
		for(int d=0; d<dim; d++){
			for(int r=0;r<nrows;r++){
				for(int c=0;c<ncols;c++){
					std::cout << pixels[0][r][c] <<  " " ;
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
	}

};

#endif