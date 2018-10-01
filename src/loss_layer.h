//
// Created by root on 9/15/18.
//

#ifndef MY_CAFFE_LOSS_LAYER_H
#define MY_CAFFE_LOSS_LAYER_H

#include <vector>
#include "blob.h"
#include "layer.h"
#include "caffe.pb.h"

namespace caffe{

    const float kLOG_THRESHOLD = 1e-20;

    //caffe实现了大量loss function，它们的父类都是 LossLayer
    template <typename Dtype>
    class LossLayer : public Layer<Dtype> {
    public:
        explicit LossLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}

        virtual void LayerSetUp(
                const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        virtual void Reshape(
                const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

        virtual inline int ExactNumBottomBlobs() const { return 2; }

        virtual inline bool AutoTopBlobs() const { return true; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

        virtual inline bool AllowForceBackward(const int bottom_index) const {
            return bottom_index != 1;
        }
    };

}

#endif //MY_CAFFE_LOSS_LAYER_H
