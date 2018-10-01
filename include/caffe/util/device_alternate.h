//
// Created by root on 7/23/18.
//

#ifndef MY_CAFFE_DEVICE_ALTERNATE_H
#define MY_CAFFE_DEVICE_ALTERNATE_H

#ifdef CPU_ONLY  // CPU-only Caffe.
#include <vector>
#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."


#else  // Normal GPU + CPU Caffe.
;
#endif

#endif //MY_CAFFE_DEVICE_ALTERNATE_H
