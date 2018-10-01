//
// Created by root on 9/15/18.
//

#ifndef MY_CAFFE_BASE_DATA_LAYER_H
#define MY_CAFFE_BASE_DATA_LAYER_H

#include <vector>
#include "layer.h"
#include "blob.h"
#include "caffe.pb.h"
#include "data_transformer.h"
#include "internal_thread.h"
#include "blocking_queue.h"

namespace caffe{
    template <typename Dtype>
    class BaseDataLayer : public Layer<Dtype> {
    public:
        explicit BaseDataLayer(const LayerParameter& param);
        // LayerSetUp: implements common data layer setup functionality, and calls
        // DataLayerSetUp to do special data layer setup for individual layer types.
        // This method may not be overridden except by the BasePrefetchingDataLayer.
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
        // Data layers should be shared by multiple solvers in parallel
        virtual inline bool ShareInParallel() const { return true; }
        virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {}
        // Data layers have no bottoms, so reshaping is trivial.
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {}

        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

    protected:
        TransformationParameter transform_param_;
        shared_ptr<DataTransformer<Dtype> > data_transformer_;
        bool output_labels_;
    };

    template <typename Dtype>
    class Batch {
    public:
        Blob<Dtype> data_, label_;
    };


    template <typename Dtype>
    class BasePrefetchingDataLayer :
            public BaseDataLayer<Dtype>, public InternalThread {
    public:
        explicit BasePrefetchingDataLayer(const LayerParameter& param);

        // LayerSetUp: implements common data layer setup functionality, and calls
        // DataLayerSetUp to do special data layer setup for individual layer types.
        // This method may not be overridden.
        void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);

        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);

        // Prefetches batches (asynchronously if to GPU memory)
        static const int PREFETCH_COUNT = 3;

    protected:
        virtual void InternalThreadEntry();
        virtual void load_batch(Batch<Dtype>* batch) = 0;

        Batch<Dtype> prefetch_[PREFETCH_COUNT];
        BlockingQueue<Batch<Dtype>*> prefetch_free_;
        BlockingQueue<Batch<Dtype>*> prefetch_full_;

        Blob<Dtype> transformed_data_;
    };



}

#endif //MY_CAFFE_BASE_DATA_LAYER_H
