//
// Created by root on 7/23/18.
//

#ifndef MY_CAFFE_SYNCEDMEM_H
#define MY_CAFFE_SYNCEDMEM_H

#include <cstdlib>
#include "common.h"


namespace caffe{
    inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
        *ptr = malloc(size);
        *use_cuda = false;
        CHECK(*ptr) << "host allocation of size " << size << " failed";
    }

    inline void CaffeFreeHost(void* ptr, bool use_cuda) {
        free(ptr);
    }

    class SyncedMemory {

    public:
            SyncedMemory()
                    : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
                      own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
                      gpu_device_(-1) {}
            explicit SyncedMemory(size_t size)
                    : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
                      own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
                      gpu_device_(-1) {}
            ~SyncedMemory();
            const void* cpu_data();
            void set_cpu_data(void* data);
            const void* gpu_data();
            void set_gpu_data(void* data);
            void* mutable_cpu_data();
            void* mutable_gpu_data();
            enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
            SyncedHead head() { return head_; }
            size_t size() { return size_; }

        private:
            void to_cpu(); //数据由显存同步到内存
            void to_gpu(); //数据由内存同步到显存
            void* cpu_ptr_;//内存指针
            void* gpu_ptr_;//显存指针
            size_t size_;  //数据大小
            SyncedHead head_;//当前数据状态，UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED
            bool own_cpu_data_;
            bool cpu_malloc_use_cuda_;
            bool own_gpu_data_;
            int gpu_device_;

            DISABLE_COPY_AND_ASSIGN(SyncedMemory);

    };

}

#endif //MY_CAFFE_SYNCEDMEM_H
