//
//  Cube_impl.hxx
//  moka
//
//  Created by Ce Zhang on 1/11/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_Cube_impl_hxx
#define moka_Cube_impl_hxx

template<typename T, LayoutType LAYOUT>
T * Cube<T, LAYOUT>::logical_get(size_t r, size_t c, size_t d, size_t b){
#ifdef _DO_ASSERT
    assert(r<R); assert(c<C); assert(d<D); assert(b<B);
#endif
    return LogicalFetcher<LAYOUT>::logical_get(*this, r,c,b,d);
};

template<typename T, LayoutType LAYOUT>
T * Cube<T, LAYOUT>::physical_get_RCslice(size_t d, size_t b){
#ifdef _DO_ASSERT
    assert(d<D); assert(b<B);
#endif
    return PhysicalFetcher<LAYOUT>::physical_get_RCslice(*this, d, b);
}

template<typename T, LayoutType LAYOUT>
Cube<T, LAYOUT>::Cube(void * _p_data, size_t _R, size_t _C, size_t _D, size_t _B) :
p_data(reinterpret_cast<T*>(_p_data)),
n_elements(_R*_C*_D*_B),
R(_R), C(_C), D(_D), B(_B),
own_data(false){}


template<typename T, LayoutType LAYOUT>
Cube<T, LAYOUT>::Cube(size_t _R, size_t _C, size_t _D, size_t _B) :
p_data((T*) malloc(sizeof(T)*_R*_C*_D*_B)), // TODO: change to 32byte align
n_elements(_R*_C*_D*_B),
R(_R), C(_C), D(_D), B(_B),
own_data(true){}


template<typename T, LayoutType LAYOUT>
Cube<T, LAYOUT>::~Cube(){
    if(own_data){
        free(p_data);
    }
}

template<typename T, LayoutType LAYOUT>
void Cube<T, LAYOUT>::logical_print(){
    for(size_t ib=0;ib<B;ib++){
        for(size_t id=0;id<D;id++){
            std::cout << "BATCH " << ib << " DEPTH " << id << std::endl;
            for(size_t ir=0;ir<R;ir++){
                std::cout << "    " ;
                for(size_t ic=0;ic<C;ic++){
                    std::cout << *logical_get(ir, ic, id, ib) << " ";
                }
                std::cout << std::endl;
            }
        }
    }
}

template<typename T, LayoutType LAYOUT>
template<typename TYPECONSTRAINT>
T* Cube<T,LAYOUT>::LogicalFetcher<Layout_RCDB, TYPECONSTRAINT>::logical_get(const Cube<T, LAYOUT>& cube, size_t r, size_t c, size_t d, size_t b){
    __builtin_prefetch((const void*)&cube.p_data[r*cube.C*cube.D*cube.B + c*cube.D*cube.B + d*cube.B + b],0,0);
    return &cube.p_data[r + c*cube.R + d*cube.R*cube.C + b*cube.R*cube.C*cube.D];
}


template<typename T, LayoutType LAYOUT>
template<typename TYPECONSTRAINT>
T* Cube<T,LAYOUT>::LogicalFetcher<Layout_BDRC, TYPECONSTRAINT>::logical_get(const Cube<T, LAYOUT>& cube, size_t r, size_t c, size_t d, size_t b){
    return &cube.p_data[b + d*cube.B + r*cube.B*cube.D + c*cube.B*cube.D*cube.R];
}

template<typename T, LayoutType LAYOUT>
template<typename TYPECONSTRAINT>
T* Cube<T,LAYOUT>::PhysicalFetcher<Layout_RCDB, TYPECONSTRAINT>::physical_get_RCslice(const Cube<T, LAYOUT>& cube, size_t d, size_t b){
    return &cube.p_data[d*cube.R*cube.C + b*cube.R*cube.C*cube.D];
}

#endif

