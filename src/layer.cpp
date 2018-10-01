//
// Created by root on 9/10/18.
//

#include <boost/thread.hpp>
#include "layer.h"

namespace caffe{

    template<typename Dtype>
    void Layer<Dtype>::InitMutex() {
        forward_mutex_.reset(new boost::mutex());
    }

    template <typename Dtype>
    void Layer<Dtype>::Lock() {
        if (IsShared()) {
            forward_mutex_->lock();
        }
    }

    template <typename Dtype>
    void Layer<Dtype>::Unlock() {
        if (IsShared()) {
            forward_mutex_->unlock();
        }
    }
    
    //模板显示实例化
    INSTANTIATE_CLASS(Layer);
    
}
