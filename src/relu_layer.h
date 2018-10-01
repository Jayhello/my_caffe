//
// Created by root on 9/12/18.
//

#ifndef MY_CAFFE_RELU_LAYER_H
#define MY_CAFFE_RELU_LAYER_H

#include "neuron_layer.h"

namespace caffe{
    template <typename Dtype>
    class ReLULayer : public NeuronLayer<Dtype> {
    public:
        explicit ReLULayer(const LayerParameter& param)
                : NeuronLayer<Dtype>(param) {}

        virtual inline const char* type() const { return "ReLU"; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top){
            Forward_cpu(bottom, top);
        }

        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
        {
            Backward_cpu(top, propagate_down, bottom);
        }
        
    };
    
}

#endif //MY_CAFFE_RELU_LAYER_H
