//
// Created by root on 7/23/18.
//

#include <iostream>
#include "caffe.h"

namespace caffe{

    template <typename T>
    void print_t(T t){
        std::cout<<"print_t T: "<<t<<std::endl;
    }
    
    template
    void print_t<int>(int t);
    
//    template <>
//    void print_t<int>(int t){
//        std::cout<<"print_t int: "<<t<<std::endl;
//    }
//
//    template <>
//    void print_t<float>(float t){
//        std::cout<<"print_t float: "<<t<<std::endl;
//    }


    template <typename T>
    void caffe<T>::test_print() {
        std::cout<<"test_print function in caffe.cpp"<<std::endl;
    }

    template class caffe<int>;
}