//
// Created by root on 9/12/18.
//

#include "neuron_layer.h"

namespace caffe{

    template <typename Dtype>
    void NeuronLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
        top[0]->ReshapeLike(*bottom[0]);
    }

    INSTANTIATE_CLASS(NeuronLayer);
    
}