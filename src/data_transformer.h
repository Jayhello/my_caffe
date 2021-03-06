//
// Created by root on 9/15/18.
//

#ifndef MY_CAFFE_DATA_TRANSFORMER_H
#define MY_CAFFE_DATA_TRANSFORMER_H
#include <vector>
#include "blob.h"
#include "common.h"
#include "caffe.pb.h"
#include <opencv2/core/core.hpp>

namespace caffe{
    /**
     * @brief Applies common transformations to the input data, such as
     * scaling, mirroring, substracting the image mean...
     */

    template <typename Dtype>
    class DataTransformer {
    public:
        explicit DataTransformer(const TransformationParameter& param, Phase phase);
        virtual ~DataTransformer() {}

        /**
         * @brief Initialize the Random number generations if needed by the
         *    transformation.
         */
        void InitRand();

        /**
         * @brief Applies the transformation defined in the data layer's
         * transform_param block to the data.
         *
         * @param datum
         *    Datum containing the data to be transformed.
         * @param transformed_blob
         *    This is destination blob. It can be part of top blob's data if
         *    set_cpu_data() is used. See data_layer.cpp for an example.
         */
        void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);

        /**
         * @brief Applies the transformation defined in the data layer's
         * transform_param block to a vector of Datum.
         *
         * @param datum_vector
         *    A vector of Datum containing the data to be transformed.
         * @param transformed_blob
         *    This is destination blob. It can be part of top blob's data if
         *    set_cpu_data() is used. See memory_layer.cpp for an example.
         */
        void Transform(const vector<Datum> & datum_vector,
                       Blob<Dtype>* transformed_blob);

        void Transform(const vector<cv::Mat> & mat_vector,
                       Blob<Dtype>* transformed_blob);

        void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob);

        /**
         * @brief Applies the same transformation defined in the data layer's
         * transform_param block to all the num images in a input_blob.
         *
         * @param input_blob
         *    A Blob containing the data to be transformed. It applies the same
         *    transformation to all the num images in the blob.
         * @param transformed_blob
         *    This is destination blob, it will contain as many images as the
         *    input blob. It can be part of top blob's data.
         */
        void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);

        /**
         * @brief Infers the shape of transformed_blob will have when
         *    the transformation is applied to the data.
         *
         * @param datum
         *    Datum containing the data to be transformed.
         */
        vector<int> InferBlobShape(const Datum& datum);

        vector<int> InferBlobShape(const vector<Datum> & datum_vector);

        vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);
        vector<int> InferBlobShape(const cv::Mat& cv_img);

    protected:
        /**
        * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
        *
        * @param n
        *    The upperbound (exclusive) value of the random number.
        * @return
        *    A uniformly random integer value from ({0, 1, ..., n-1}).
        */
        virtual int Rand(int n);
        void Transform(const Datum& datum, Dtype* transformed_data);

    protected:
        // Tranformation parameters
        TransformationParameter param_;

        shared_ptr<Caffe::RNG> rng_;
        Phase phase_;
        Blob<Dtype> data_mean_;
        vector<Dtype> mean_values_;

    };

}

#endif //MY_CAFFE_DATA_TRANSFORMER_H
