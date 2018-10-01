//
// Created by root on 9/11/18.
//

#ifndef MY_CAFFE_RNG_H
#define MY_CAFFE_RNG_H

#include <algorithm>
#include <iterator>
#include <boost/random.hpp>
#include "common.h"

namespace caffe{

    typedef boost::mt19937 rng_t;

    inline rng_t* caffe_rng() {
//        return static_cast<caffe::rng_t*>(Caffe::rng_stream().generator());
        return static_cast<rng_t*>(Caffe::rng_stream().generator());
    }

// Fisher–Yates algorithm
    template <class RandomAccessIterator, class RandomGenerator>
    inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end,
                        RandomGenerator* gen) {
        typedef typename std::iterator_traits<RandomAccessIterator>::difference_type
                difference_type;
        typedef typename boost::uniform_int<difference_type> dist_type;

        difference_type length = std::distance(begin, end);
        if (length <= 0) return;

        for (difference_type i = length - 1; i > 0; --i) {
            dist_type dist(0, i);
            std::iter_swap(begin + i, begin + dist(*gen));
        }
    }

    template <class RandomAccessIterator>
    inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end) {
        shuffle(begin, end, caffe_rng());
    }

}

#endif //MY_CAFFE_RNG_H
