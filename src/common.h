//
// Created by root on 7/23/18.
//

#ifndef MY_CAFFE_COMMON_H
#define MY_CAFFE_COMMON_H

#include <glog/logging.h>
#include <boost/shared_ptr.hpp>
#include <map>
#include <set>


// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>


#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"


namespace caffe{

    // We will use the boost shared_ptr instead of the new C++11 one mainly
    // because cuda does not work (at least now) well with C++11 features.
    using boost::shared_ptr;

    // Common functions and classes from std that caffe often uses.
    using std::fstream;
    using std::ios;
    using std::isnan;
    using std::isinf;
    using std::iterator;
    using std::make_pair;
    using std::map;
    using std::ostringstream;
    using std::pair;
    using std::set;
    using std::string;
    using std::stringstream;
    using std::vector;

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
    class Caffe {
    public:
        //Brew就是CPU，GPU的枚举类型
        enum Brew { CPU, GPU };
        // This random number generator facade hides boost and CUDA rng
        // implementation from one another (for cross-platform compatibility).
        class RNG {
        public:
            RNG(); //利用系统的熵池或者时间来初始化RNG内部的generator_
            explicit RNG(unsigned int seed);
            explicit RNG(const RNG&);
            RNG& operator=(const RNG&);
            void* generator();
        private:
            class Generator;
            shared_ptr<Generator> generator_;
        };

        // Getters for boost rng, curand, and cublas handles
        inline static RNG& rng_stream() {
          if (!Get().random_generator_) {
            Get().random_generator_.reset(new RNG());
          }
          return *(Get().random_generator_);
        }

    public:

        //Get函数利用Boost的局部线程存储功能实现
        static Caffe& Get();

        ~Caffe();

    public:
        // Returns the mode: running on CPU or GPU.
        inline static Brew mode() { return Get().mode_; }

        inline static void set_mode(Brew mode) { Get().mode_ = mode; }
        static void set_random_seed(const unsigned int seed);

        static void SetDevice(const int device_id);
        // Prints the current GPU status.
        static void DeviceQuery();
        // Check if specified device is available
        static bool CheckDevice(const int device_id);
        static int FindDevice(const int start_id = 0);

        // Parallel training info
        // Parallel training info
        inline static int solver_count() { return Get().solver_count_; }
        inline static void set_solver_count(int val) { Get().solver_count_ = val; }
        inline static bool root_solver() { return Get().root_solver_; }
        inline static void set_root_solver(bool val) { Get().root_solver_ = val; }

    protected:
        shared_ptr<RNG> random_generator_;

        Brew mode_;
        int solver_count_;
        bool root_solver_;

    private:
        //实现中构造函数被声明为私有方法，这样从根本上杜绝外部使用构造函数生成新的实例，
        Caffe();
        //同时禁用拷贝函数与赋值操作符（声明为私有但是不提供实现）避免通过拷贝函数或赋值操作生成新实例。
        DISABLE_COPY_AND_ASSIGN(Caffe);

    };

}

#endif //MY_CAFFE_COMMON_H
