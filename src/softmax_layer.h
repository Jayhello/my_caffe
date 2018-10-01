//
// Created by root on 9/15/18.
//

#include <vector>

#include "blob.h"
#include "layer.h"
#include "caffe.pb.h"

namespace caffe{

    template <typename Dtype>
    class SoftmaxLayer : public Layer<Dtype> {
    public:
        explicit SoftmaxLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}

        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "Softmax"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

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
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
            Backward_cpu(top, propagate_down, bottom);
        }

        int outer_num_;
        int inner_num_;
        int softmax_axis_;
        /// sum_multiplier is used to carry out sum using BLAS
        Blob<Dtype> sum_multiplier_;
        /// scale is an intermediate Blob to hold temporary results.
        Blob<Dtype> scale_;

    };

}