//
// Created by root on 9/18/18.
//

#include "base_data_layer.h"
#include "device_alternate.h"

namespace caffe{

    template <typename Dtype>
    BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
            : Layer<Dtype>(param),
              transform_param_(param.transform_param()) {
    }

    template <typename Dtype>
    void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
        if (top.size() == 1) {
            output_labels_ = false;
        } else {
            output_labels_ = true;
        }
        data_transformer_.reset(
                new DataTransformer<Dtype>(transform_param_, this->phase_));
        data_transformer_->InitRand();
        // The subclasses should setup the size of bottom and top
        DataLayerSetUp(bottom, top);
    }

    template <typename Dtype>
    BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
            const LayerParameter& param)
            : BaseDataLayer<Dtype>(param),
              prefetch_free_(), prefetch_full_() {
        for (int i = 0; i < PREFETCH_COUNT; ++i) {
            prefetch_free_.push(&prefetch_[i]);
        }
    }

    template <typename Dtype>
    void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
        // Before starting the prefetch thread, we make cpu_data and gpu_data
        // calls so that the prefetch thread does not accidentally make simultaneous
        // cudaMalloc calls when the main thread is running. In some GPUs this
        // seems to cause failures if we do not so.
        for (int i = 0; i < PREFETCH_COUNT; ++i) {
            prefetch_[i].data_.mutable_cpu_data();
            if (this->output_labels_) {
                prefetch_[i].label_.mutable_cpu_data();
            }
        }
#ifndef CPU_ONLY
        if (Caffe::mode() == Caffe::GPU) {
            for (int i = 0; i < PREFETCH_COUNT; ++i) {
                prefetch_[i].data_.mutable_gpu_data();
                if (this->output_labels_) {
                    prefetch_[i].label_.mutable_gpu_data();
                }
            }
        }
#endif
        DLOG(INFO) << "Initializing prefetch";
        this->data_transformer_->InitRand();
        StartInternalThread();
        DLOG(INFO) << "Prefetch initialized.";
    }

    template <typename Dtype>
    void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
        try {
            while (!must_stop()) {
                Batch<Dtype> *batch = prefetch_free_.pop();
                load_batch(batch);

                prefetch_full_.push(batch);
            }
        } catch (boost::thread_interrupted&) {
            // Interrupted exception is expected on shutdown
        }

    }

    template <typename Dtype>
    void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

        Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
        // Reshape to loaded data.
        top[0]->ReshapeLike(batch->data_);

        // Copy the data
        caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
                   top[0]->mutable_cpu_data());

//        DLOG(INFO) << "Prefetch copied";
        if (this->output_labels_) {
            // Reshape to loaded labels.
            top[1]->ReshapeLike(batch->label_);
            // Copy the labels.
            caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
                       top[1]->mutable_cpu_data());
        }

        prefetch_free_.push(batch);
    }

    STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
    
    INSTANTIATE_CLASS(BaseDataLayer);
    INSTANTIATE_CLASS(BasePrefetchingDataLayer);    
}
