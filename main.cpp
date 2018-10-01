#include "test_blob.h"
#include <iostream>
#include <memory>

using std::cout;
using std::endl;
using std::string;

using std::shared_ptr;
using std::weak_ptr;


void test(){
    int two[] = {1, 2, 3};
    two = {1, 2, 4}; // error

}


int main(int argc, char** argv) {
    test();

//    caffe::test_net();
//    caffe::test_mnist();

//    caffe::test_prototxt();
//    caffe::test_net();
//    caffe::test_solver();
//    caffe::test_blob_filter();
//    test_blob_1();
//    test_blob_2();
//    test_blob_3();

//    std::cout << "Hello, World!" << std::endl;
    return 0;
}