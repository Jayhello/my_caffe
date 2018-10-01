//
// Created by root on 7/23/18.
//

#ifndef MY_CAFFE_DEVICE_ALTERNATE_H
#define MY_CAFFE_DEVICE_ALTERNATE_H

//#ifdef CPU_ONLY  // CPU-only Caffe.
#include <vector>
#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."

#define STUB_GPU(classname) \
template <typename Dtype> \
void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \
template <typename Dtype> \
void classname<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

#define STUB_GPU_FORWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \

#define STUB_GPU_BACKWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

//#else  // Normal GPU + CPU Caffe.
//;
//#endif

#endif //MY_CAFFE_DEVICE_ALTERNATE_H
