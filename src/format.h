//
// Created by root on 9/13/18.
//

#ifndef MY_CAFFE_FORMAT_H
#define MY_CAFFE_FORMAT_H

#include <iomanip>  // NOLINT(readability/streams)
#include <sstream>  // NOLINT(readability/streams)
#include <string>

namespace caffe{
    inline std::string format_int(int n, int numberOfLeadingZeros = 0 ) {
        std::ostringstream s;
        s << std::setw(numberOfLeadingZeros)<<std::setfill('0')<<n;
        return s.str();
    }

}

#endif //MY_CAFFE_FORMAT_H
