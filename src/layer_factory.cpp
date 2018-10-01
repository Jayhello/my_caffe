//
// Created by root on 9/14/18.
//

#include "layer.h"
#include "layer_factory.h"
#include "relu_layer.h"
#include "inner_product_layer.h"
#include "softmax_layer.h"


namespace caffe{

// Get relu layer according to engine.
    template <typename Dtype>
    shared_ptr<Layer<Dtype> > GetReLULayer(const LayerParameter& param) {
        ReLUParameter_Engine engine = param.relu_param().engine();
        if (engine == ReLUParameter_Engine_DEFAULT) {
            engine = ReLUParameter_Engine_CAFFE;
#ifdef USE_CUDNN
            engine = ReLUParameter_Engine_CUDNN;
#endif
        }
        if (engine == ReLUParameter_Engine_CAFFE) {
            return shared_ptr<Layer<Dtype> >(new ReLULayer<Dtype>(param));
#ifdef USE_CUDNN
            } else if (engine == ReLUParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNReLULayer<Dtype>(param));
#endif
        } else {
            LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
        }
    }

    REGISTER_LAYER_CREATOR(ReLU, GetReLULayer);

// Get softmax layer according to engine.
    template <typename Dtype>
    shared_ptr<Layer<Dtype> > GetSoftmaxLayer(const LayerParameter& param) {
        SoftmaxParameter_Engine engine = param.softmax_param().engine();
        if (engine == SoftmaxParameter_Engine_DEFAULT) {
            engine = SoftmaxParameter_Engine_CAFFE;
#ifdef USE_CUDNN
            engine = SoftmaxParameter_Engine_CUDNN;
#endif
        }
        if (engine == SoftmaxParameter_Engine_CAFFE) {
            return shared_ptr<Layer<Dtype> >(new SoftmaxLayer<Dtype>(param));
#ifdef USE_CUDNN
            } else if (engine == SoftmaxParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNSoftmaxLayer<Dtype>(param));
#endif
        } else {
            LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
        }
    }

    REGISTER_LAYER_CREATOR(Softmax, GetSoftmaxLayer);

}