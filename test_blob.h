//
// Created by root on 9/5/18.
//

#ifndef MY_CAFFE_TEST_BLOB_H
#define MY_CAFFE_TEST_BLOB_H

#include <cstdint>
#include <string>

using std::string;

namespace caffe{

/*
 * simple test of Blob
 * set value to it and print the value
 */
void test_blob_1();


/*
 * read image from opencv and set it value to Blob
 * operator Blob(draw a line) and set it back to cv Mat
 */
void test_blob_2();

/*
 * test shape size 2 blob
 */

void test_blob_3();

/*
 * test filter blob
 */
void test_blob_filter();

    void test_prototxt();

void test_net();

    void test_solver();

    void test_mnist();

    uint32_t swap_endian(uint32_t val);
    void convert_dataset(const char* image_filename, const char* label_filename,
                         const char* db_path, const string& db_backend = "lmdb");

    void read_mnist_cv(const char* image_filename, const char* label_filename);

    void test_data_layer();
}

#endif //MY_CAFFE_TEST_BLOB_H
