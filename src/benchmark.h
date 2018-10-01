//
// Created by root on 9/18/18.
//

#ifndef MY_CAFFE_BENCHMARK_H
#define MY_CAFFE_BENCHMARK_H


#include <boost/date_time/posix_time/ptime.hpp>

namespace caffe{

    class Timer {
    public:
        Timer();
        virtual ~Timer();
        virtual void Start();
        virtual void Stop();
        virtual float MilliSeconds();
        virtual float MicroSeconds();
        virtual float Seconds();

        inline bool initted() { return initted_; }
        inline bool running() { return running_; }
        inline bool has_run_at_least_once() { return has_run_at_least_once_; }

    protected:
        void Init();

        bool initted_;
        bool running_;
        bool has_run_at_least_once_;
#ifndef CPU_ONLY
        cudaEvent_t start_gpu_;
        cudaEvent_t stop_gpu_;
#endif
        boost::posix_time::ptime start_cpu_;
        boost::posix_time::ptime stop_cpu_;
        float elapsed_milliseconds_;
        float elapsed_microseconds_;
    };

    class CPUTimer : public Timer {
    public:
        explicit CPUTimer();
        virtual ~CPUTimer() {}
        virtual void Start();
        virtual void Stop();
        virtual float MilliSeconds();
        virtual float MicroSeconds();
    };    
    
}

#endif //MY_CAFFE_BENCHMARK_H
