//
// Created by root on 9/19/18.
//

#ifndef MY_CAFFE_SPLIT_LAYER_H
#define MY_CAFFE_SPLIT_LAYER_H

#include "layer.h"


namespace caffe{

/**
 * @brief Creates a "split" path in the network by copying the bottom Blob
 *        into multiple top Blob%s to be used by multiple consuming layers.
 *  用于一输入对多输出的场合（对blob）
 */

    template <typename Dtype>
    class SplitLayer : public Layer<Dtype> {
    public:
        explicit SplitLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "Split"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int MinTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        int count_;
    };


}

#endif //MY_CAFFE_SPLIT_LAYER_H
