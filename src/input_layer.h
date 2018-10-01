//
// Created by root on 9/14/18.
//

#ifndef MY_CAFFE_INPUT_LAYER_H
#define MY_CAFFE_INPUT_LAYER_H

#include <vector>
#include "layer.h"
#include "blob.h"
#include "caffe.pb.h"
#include "layer_factory.h"

namespace caffe{

    template <typename Dtype>
    class InputLayer : public Layer<Dtype> {
    public:
        explicit InputLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}

        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        // Data layers should be shared by multiple solvers in parallel
        virtual inline bool ShareInParallel() const { return true; }
        // Data layers have no bottoms, so reshaping is trivial.
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top) {}

        virtual inline const char* type() const { return "Input"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int MinTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {}
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

        }
        
    };


    template <typename Dtype>
    void InputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                                       const vector<Blob<Dtype>*>& top) {
        const int num_top = top.size();
        const InputParameter& param = this->layer_param_.input_param();
        const int num_shape = param.shape_size();
        CHECK(num_shape == 0 || num_shape == 1 || num_shape == num_top)
        << "Must specify 'shape' once, once per top blob, or not at all: "
        << num_top << " tops vs. " << num_shape << " shapes.";
    
        if (num_shape > 0) {
            for (int i = 0; i < num_top; ++i) {
                const int shape_index = (param.shape_size() == 1) ? 0 : i;
                top[i]->Reshape(param.shape(shape_index));
            }
        }
    }

INSTANTIATE_CLASS(InputLayer);
REGISTER_LAYER_CLASS(Input);
    
}

#endif //MY_CAFFE_INPUT_LAYER_H
