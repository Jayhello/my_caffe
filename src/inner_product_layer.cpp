//
// Created by root on 9/11/18.
//
#include "inner_product_layer.h"
#include "filter.h"
#include "math_functions.h"
#include "layer_factory.h"
#include "device_alternate.h"

namespace caffe {
    template<typename Dtype>
    void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                              const vector<Blob<Dtype> *> &top) {

        // 根据层参数设置基本信息
        const int num_output = this->layer_param_.inner_product_param().num_output();
        // whether to have bias terms
        bias_term_ = this->layer_param_.inner_product_param().bias_term();
        transpose_ = this->layer_param_.inner_product_param().transpose();

        N_ = num_output;

        const int axis = bottom[0]->CanonicalAxisIndex(
                this->layer_param_.inner_product_param().axis());
        // Dimensions starting from "axis" are "flattened" into a single
        // length K_ vector. For examples, if bottom[0]'s shape is (N, C, H, W),
        // and axis == 1, N inner products with dimension CHW are performed.
        // K_ = C*H*W
        K_ = bottom[0]->count(axis);

        // Check if we need to set up the weights
        if (this->blobs_.size() > 0) {
            LOG(INFO) << "Skipping parameter initialization";
        } else {
            if (bias_term_) {
                this->blobs_.resize(2);
            } else {
                this->blobs_.resize(1);
            }

            //TODO shape size is 2 not 4
            // Initialize the weights
            // 权值初始化
            vector<int> weight_shape(2);
            if (transpose_) {
                weight_shape[0] = K_;
                weight_shape[1] = N_;
            } else {
                // 权值维数[N_,k_]
                weight_shape[0] = N_;
                weight_shape[1] = K_;
            }

            this->blobs_[0].reset(new Blob<Dtype>(weight_shape));

            // get filter type and then fill weights
            shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                    this->layer_param_.inner_product_param().weight_filler()));

            // fill the weights
            weight_filler->Fill(this->blobs_[0].get());

            // If necessary, intiialize and fill the bias term
            if (bias_term_) {
                vector<int> bias_shape(1, N_);
                this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
                shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                        this->layer_param_.inner_product_param().bias_filler()));
                bias_filler->Fill(this->blobs_[1].get());
            }

        } // end of parameter initialization

        // 设置每个参数是否需要反向传播
        this->param_propagate_down_.resize(this->blobs_.size(), true);
    }

    template <typename Dtype>
    void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {

        // Figure out the dimensions
        const int axis = bottom[0]->CanonicalAxisIndex(
                this->layer_param_.inner_product_param().axis());

        const int new_K = bottom[0]->count(axis);
        CHECK_EQ(K_, new_K)
            << "Input size incompatible with inner product parameters.";

        // The first "axis" dimensions are independent inner products; the total
        // number of these is M_, the product over these dimensions.
        // M_就是batch size N
        M_ = bottom[0]->count(0, axis);

        // The top shape will be the bottom shape with the flattened axes dropped,
        // and replaced by a single axis with dimension num_output (N_).
        // top_shape:[N,C,H,W]
        vector<int> top_shape = bottom[0]->shape();

        // top_shape:[N,C]
        top_shape.resize(axis + 1);
        // top_shape:[N,N_]
        top_shape[axis] = N_;
        // 设置top的形状大小
        top[0]->Reshape(top_shape);

        // Set up the bias multiplier
        if (bias_term_) {
            vector<int> bias_shape(1, M_);
            bias_multiplier_.Reshape(bias_shape);
            caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
        }
    }

    template<typename Dtype>
    void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                               const vector<Blob<Dtype> *> &top) {

        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        const Dtype* weight = this->blobs_[0]->cpu_data();

        // top = bottom * weight + bias (option)
        caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
                              M_, N_, K_, (Dtype)1.,
                              bottom_data, weight, (Dtype)0., top_data);
        if (bias_term_) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                                  bias_multiplier_.cpu_data(),
                                  this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
        }

    }

    template <typename Dtype>
    void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                const vector<bool>& propagate_down,
                                                const vector<Blob<Dtype>*>& bottom) {
        if (this->param_propagate_down_[0]) {
            const Dtype* top_diff = top[0]->cpu_diff();
            const Dtype* bottom_data = bottom[0]->cpu_data();
            // Gradient with respect to weight
            if (transpose_) {
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                      K_, N_, M_,
                                      (Dtype)1., bottom_data, top_diff,
                                      (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
            } else {
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                      N_, K_, M_,
                                      (Dtype)1., top_diff, bottom_data,
                                      (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
            }
        }
        if (bias_term_ && this->param_propagate_down_[1]) {
            const Dtype* top_diff = top[0]->cpu_diff();
            // Gradient with respect to bias
            caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
                                  bias_multiplier_.cpu_data(), (Dtype)1.,
                                  this->blobs_[1]->mutable_cpu_diff());
        }
        if (propagate_down[0]) {
            const Dtype* top_diff = top[0]->cpu_diff();
            // Gradient with respect to bottom data
            if (transpose_) {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                                      M_, K_, N_,
                                      (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
                                      (Dtype)0., bottom[0]->mutable_cpu_diff());
            } else {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                                      M_, K_, N_,
                                      (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
                                      (Dtype)0., bottom[0]->mutable_cpu_diff());
            }
        }
    }

#ifdef CPU_ONLY
//    STUB_GPU(InnerProductLayer);
#endif

    INSTANTIATE_CLASS(InnerProductLayer);
    REGISTER_LAYER_CLASS(InnerProduct);

}



