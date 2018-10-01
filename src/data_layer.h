//
// Created by root on 9/18/18.
//

#ifndef MY_CAFFE_DATA_LAYER_H
#define MY_CAFFE_DATA_LAYER_H

#include <vector>
#include "layer.h"
#include "blob.h"
#include "caffe.pb.h"
#include "base_data_layer.h"
#include "data_reader.h"

namespace caffe{

/*
原始数据的输入层，处于整个网络的最底层，它可以
从数据库leveldb、 lmdb中读取数据，也可以直接从内存中读取，还
可以从hdf5，甚至是原始的图像读入数据。 作为网络的最底层，主
要实现数据格式的转换
 */

    template <typename Dtype>
    class DataLayer : public BasePrefetchingDataLayer<Dtype> {
    public:
        explicit DataLayer(const LayerParameter& param);
        virtual ~DataLayer();
        virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top);
        // DataLayer uses DataReader instead for sharing for parallelism
        virtual inline bool ShareInParallel() const { return false; }
        virtual inline const char* type() const { return "Data"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlobs() const { return 2; }

    protected:
        virtual void load_batch(Batch<Dtype>* batch);

        DataReader reader_;

    };

}

#endif //MY_CAFFE_DATA_LAYER_H
