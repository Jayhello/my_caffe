//
// Created by root on 9/15/18.
//

#ifndef MY_CAFFE_SOFTMAX_LOSS_LAYER_H
#define MY_CAFFE_SOFTMAX_LOSS_LAYER_H

#include <cfloat>
#include "loss_layer.h"
#include "layer_factory.h"
#include "device_alternate.h"

namespace caffe{

    template <typename Dtype>
    class SoftmaxWithLossLayer : public LossLayer<Dtype> {
        public:
        explicit SoftmaxWithLossLayer(const LayerParameter& param)
                : LossLayer<Dtype>(param) {}

        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "SoftmaxWithLoss"; }
        virtual inline int ExactNumTopBlobs() const { return -1; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlobs() const { return 2; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);

        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        /// Read the normalization mode parameter and compute the normalizer based
        /// on the blob size.  If normalization_mode is VALID, the count of valid
        /// outputs will be read from valid_count, unless it is -1 in which case
        /// all outputs are assumed to be valid.
        virtual Dtype get_normalizer(
                LossParameter_NormalizationMode normalization_mode, int valid_count);

        /// The internal SoftmaxLayer used to map predictions to a distribution.
        shared_ptr<Layer<Dtype> > softmax_layer_;
        /// prob stores the output probability predictions from the SoftmaxLayer.
        Blob<Dtype> prob_;
        /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
        vector<Blob<Dtype>*> softmax_bottom_vec_;
        /// top vector holder used in call to the underlying SoftmaxLayer::Forward
        vector<Blob<Dtype>*> softmax_top_vec_;
        /// Whether to ignore instances with a certain label.
        bool has_ignore_label_;
        /// The label indicating that an instance should be ignored.
        int ignore_label_;
        /// How to normalize the output loss.
        LossParameter_NormalizationMode normalization_;

        int softmax_axis_, outer_num_, inner_num_;
    };


}

#endif //MY_CAFFE_SOFTMAX_LOSS_LAYER_H
