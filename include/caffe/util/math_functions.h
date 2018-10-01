//
// Created by root on 7/23/18.
//

#ifndef MY_CAFFE_MATH_FUNCTIONS_H
#define MY_CAFFE_MATH_FUNCTIONS_H


#include <cstring>

namespace caffe{

    inline void caffe_memset(const size_t N, const int alpha, void* X) {
        memset(X, alpha, N);  // NOLINT(caffe/alt_fn)
    }

    template <typename Dtype>
    void caffe_scal(const int N, const Dtype alpha, Dtype *X);

    template <typename Dtype>
    void caffe_axpy(const int N, const Dtype alpha, const Dtype* X,
                    Dtype* Y);

    // Returns the sum of the absolute values of the elements of vector x
    template <typename Dtype>
    Dtype caffe_cpu_asum(const int n, const Dtype* x);

    template <typename Dtype>
    Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y);

    template <typename Dtype>
    Dtype caffe_cpu_strided_dot(const int n, const Dtype* x, const int incx,
                                const Dtype* y, const int incy);

}

#endif //MY_CAFFE_MATH_FUNCTIONS_H
