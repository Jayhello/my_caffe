//
// Created by root on 9/13/18.
//

#ifndef MY_CAFFE_INSERT_SPLITS_H
#define MY_CAFFE_INSERT_SPLITS_H

#include <string>

#include "caffe.pb.h"

namespace caffe{

    // Copy NetParameters with SplitLayers added to replace any shared bottom
// blobs with unique bottom blobs provided by the SplitLayer.
    void InsertSplits(const NetParameter& param, NetParameter* param_split);

    void ConfigureSplitLayer(const string& layer_name, const string& blob_name,
                             const int blob_idx, const int split_count, const float loss_weight,
                             LayerParameter* split_layer_param);

    string SplitLayerName(const string& layer_name, const string& blob_name,
                          const int blob_idx);

    string SplitBlobName(const string& layer_name, const string& blob_name,
                         const int blob_idx, const int split_idx);
}

#endif //MY_CAFFE_INSERT_SPLITS_H
