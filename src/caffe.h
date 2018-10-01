//
// Created by root on 7/23/18.
//

#ifndef MY_CAFFE_CAFFE_H
#define MY_CAFFE_CAFFE_H

namespace caffe{

    template <typename T>
    void print_t(T t);

    template <typename T>
    class caffe{

        public:
            void test_print();

        private:
            T val;
    };

}

#endif //MY_CAFFE_CAFFE_H
