//
// Created by root on 9/11/18.
//

#ifndef MY_CAFFE_INNER_PRODUCT_LAYER_H
#define MY_CAFFE_INNER_PRODUCT_LAYER_H

#include <vector>
#include "layer.h"
#include "blob.h"
#include "caffe.pb.h"


namespace caffe{

    template <typename Dtype>
    class InnerProductLayer : public Layer<Dtype> {
    public:
        //如果父类只有有参数的构造方法，则子类必须显示调用此带参构造方法
        explicit InnerProductLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}

    public:

        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "InnerProduct"; }
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

    protected:

        int M_; // batch size
        int K_; // 输入特征长度
        int N_; // 输出神经元数量
        bool bias_term_; //是否添加偏置
        Blob<Dtype> bias_multiplier_; //偏置的乘子
        bool transpose_;  ///< if true, assume transposed weights
    };


}

#endif //MY_CAFFE_INNER_PRODUCT_LAYER_H
