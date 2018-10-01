//
// Created by root on 9/19/18.
//

#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>

#include <stdint.h>
#include <sys/stat.h>

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include <boost/scoped_ptr.hpp>

#include "caffe.pb.h"

#include "db.h"
#include "io.h"
#include "format.h"
#include "test_blob.h"
#include <iostream>
#include <opencv2/core/types_c.h>
#include <ml.h>
#include <highgui.h>
#include <cv.h>

using boost::scoped_ptr;
using std::string;
using std::cout;
using std::endl;


namespace caffe{

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void convert_dataset(const char* image_filename, const char* label_filename,
                     const char* db_path, const string& db_backend) {
    // Open files
    std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
    CHECK(image_file) << "Unable to open file " << image_filename;
    CHECK(label_file) << "Unable to open file " << label_filename;
    // Read the magic and the meta data
    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;

    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    CHECK_EQ(num_items, num_labels);
    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);


    scoped_ptr<db::DB> db(db::GetDB(db_backend));
    db->Open(db_path, db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());

    // Storing to db
    char label;
    char* pixels = new char[rows * cols];
    int count = 0;
    string value;

    Datum datum;
    datum.set_channels(1);
    datum.set_height(rows);
    datum.set_width(cols);
    LOG(INFO) << "A total of " << num_items << " items.";
    LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
    for (int item_id = 0; item_id < num_items; ++item_id) {
        image_file.read(pixels, rows * cols);
        label_file.read(&label, 1);
        datum.set_data(pixels, rows*cols);
        datum.set_label(label);
        string key_str = caffe::format_int(item_id, 8);
        datum.SerializeToString(&value);

        txn->Put(key_str, value);

        if (++count % 1000 == 0) {
            txn->Commit();
        }
    }
    // write the last batch
    if (count % 1000 != 0) {
        txn->Commit();
    }
    LOG(INFO) << "Processed " << count << " files.";
    delete[] pixels;
    db->Close();
}

    void read_mnist_cv(const char* image_filename, const char* label_filename){
        // Open files
        std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
        std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);

        // Read the magic and the meta data
        uint32_t magic;
        uint32_t num_items;
        uint32_t num_labels;
        uint32_t rows;
        uint32_t cols;

        image_file.read(reinterpret_cast<char*>(&magic), 4);
        magic = swap_endian(magic);
        if(magic != 2051){
            cout<<"Incorrect image file magic: "<<magic<<endl;
            return;
        }

        label_file.read(reinterpret_cast<char*>(&magic), 4);
        magic = swap_endian(magic);
        if(magic != 2049){
            cout<<"Incorrect image file magic: "<<magic<<endl;
            return;
        }

        image_file.read(reinterpret_cast<char*>(&num_items), 4);
        num_items = swap_endian(num_items);
        label_file.read(reinterpret_cast<char*>(&num_labels), 4);
        num_labels = swap_endian(num_labels);
        if(num_items != num_labels){
            cout<<"image file nums should equal to label num"<<endl;
            return;
        }

        image_file.read(reinterpret_cast<char*>(&rows), 4);
        rows = swap_endian(rows);
        image_file.read(reinterpret_cast<char*>(&cols), 4);
        cols = swap_endian(cols);

        cout<<"image and label num is: "<<num_items<<endl;
        cout<<"image rows: "<<rows<<", cols: "<<cols<<endl;

        Datum datum;
        datum.set_channels(1);
        datum.set_height(rows);
        datum.set_width(cols);

        char label;
        char* pixels = new char[rows * cols];

        for (int item_id = 0; item_id < num_items; ++item_id) {
            // read image pixel
            image_file.read(pixels, rows * cols);
            // read label
            label_file.read(&label, 1);
            string sLabel = std::to_string(int(label));

            datum.set_data(pixels, rows*cols);
            datum.set_label(label);
            string path = "/home/xy/caffe_analysis/my_caffe/example/" + sLabel + ".proto";
            WriteProtoToBinaryFile(datum, path);

            cout<<"lable is: "<<sLabel<<endl;
            // convert it to cv Mat, and show it
            cv::Mat image_tmp(rows,cols,CV_8UC1,pixels);
            string img_name = "/home/xy/caffe_analysis/my_caffe/example/" + sLabel + ".jpg";
            cv::imwrite(img_name, image_tmp);

            // resize bigger for showing
//            cv::resize(image_tmp, image_tmp, cv::Size(100, 100));
//            cv::imshow(sLabel, image_tmp);
//            cv::waitKey(0);
        }

        delete[] pixels;
    }

    void test_data_layer(){



    }

}