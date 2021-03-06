//
// Created by root on 9/15/18.
//

#ifndef MY_CAFFE_BLOCKING_QUEUE_H
#define MY_CAFFE_BLOCKING_QUEUE_H

#include <queue>
#include <string>
#include <boost/thread.hpp>
#include "common.h"


namespace caffe {

    template<typename T>
    class BlockingQueue {
    public:
        explicit BlockingQueue();

        void push(const T& t);

        bool try_pop(T* t);

        // This logs a message if the threads needs to be blocked
        // useful for detecting e.g. when data feeding is too slow
        T pop(const string& log_on_wait = "");

        bool try_peek(T* t);

        // Return element without removing it
        T peek();

        size_t size() const;

    protected:
        class sync;

        std::queue<T> queue_;
        shared_ptr<sync> sync_;
        DISABLE_COPY_AND_ASSIGN(BlockingQueue);
    };

}

#endif //MY_CAFFE_BLOCKING_QUEUE_H
