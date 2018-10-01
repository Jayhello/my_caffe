//
// Created by root on 9/15/18.
//

#ifndef MY_CAFFE_INTERNAL_THREAD_H
#define MY_CAFFE_INTERNAL_THREAD_H

#include "common.h"

namespace boost { class thread; }

namespace caffe{

    /*
     * 封装了pthread函数，继承的子类可以得到一个单独的线程，主要作用是在计算当前的一批数据时，在后台获取
     * 新一批的数据。
    */

    class InternalThread {
    public:
        InternalThread() : thread_() {}
        virtual ~InternalThread();

        /**
         * Caffe's thread local state will be initialized using the current
         * thread values, e.g. device id, solver index etc. The random seed
         * is initialized using caffe_rng_rand.
         */
        void StartInternalThread();

        /** Will not return until the internal thread has exited. */
        void StopInternalThread();

        bool is_started() const;

    protected:
        /* Implement this method in your subclass
        with the code you want your thread to run. */
        virtual void InternalThreadEntry() {}

        /* Should be tested when running loops to exit when requested. */
        bool must_stop();

    private:
        void entry(int device, Caffe::Brew mode, int rand_seed, int solver_count,
                   bool root_solver);

    private:
        shared_ptr<boost::thread> thread_;
    };
}

#endif //MY_CAFFE_INTERNAL_THREAD_H
