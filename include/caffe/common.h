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



}

#endif //MY_CAFFE_COMMON_H
