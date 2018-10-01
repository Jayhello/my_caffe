//
// Created by root on 9/15/18.
//

#ifndef MY_CAFFE_ACCURACY_LAYER_H
#define MY_CAFFE_ACCURACY_LAYER_H

#include <vector>
#include "layer.h"
#include "blob.h"
#include "caffe.pb.h"

namespace caffe{
    
    /**
 * @brief Computes the classification accuracy for a one-of-many
 *        classification task.
 */
    template <typename Dtype>
    class AccuracyLayer : public Layer<Dtype> {
    public:
        explicit AccuracyLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "Accuracy"; }
        virtual inline int ExactNumBottomBlobs() const { return 2; }

        // If there are two top blobs, then the second blob will contain
        // accuracies per class.
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlos() const { return 2; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);


        /// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
            for (int i = 0; i < propagate_down.size(); ++i) {
                if (propagate_down[i]) { NOT_IMPLEMENTED; }
            }
        }

        int label_axis_, outer_num_, inner_num_;

        int top_k_;

        /// Whether to ignore instances with a certain label.
        bool has_ignore_label_;
        /// The label indicating that an instance should be ignored.
        int ignore_label_;
        /// Keeps counts of the number of samples per class.
        Blob<Dtype> nums_buffer_;
    };

}

#endif //MY_CAFFE_ACCURACY_LAYER_H
