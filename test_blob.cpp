//
// Created by root on 9/5/18.
//

#include "test_blob.h"
#include <iostream>
#include "src/blob.h"
#include "src/caffe.h"
#include <opencv2/opencv.hpp>
#include <upgrade_proto.h>
#include <io.h>
#include <inner_product_layer.h>
#include <input_layer.h>
#include <data_layer.h>
#include <softmax_loss_layer.h>
#include <softmax_layer.h>
#include "src/filter.h"
#include "src/caffe.pb.h"
#include "net.h"
#include "layer_factory.h"
#include "relu_layer.h"
#include "sgd_solvers.h"


using std::vector;
using std::cout;
using std::endl;

namespace caffe{

void test_blob_1(){

    google::InitGoogleLogging("test_blob_1");
    FLAGS_alsologtostderr = 1;
//    google::SetLogDestination(google::GLOG_INFO,"./myInfo");
    LOG(INFO) << "glog: HELLO " << "ok!";


    SyncedMemory s(100);
    s.cpu_data();

    // create Blob with (N, C, H, W) -> (1, 2, 3, 4)
    std::vector<int> vecShape{1, 2, 3, 4};
    Blob<float> b(vecShape);

    // fill Blob with sequence
    float* pf = b.mutable_cpu_data();
    std::iota(pf, pf + 24, 0.0);

    // output all the sequence in Blob
    for (int i = 0; i < b.count(); ++i) {
        std::cout << *(pf + i)<<" ";
    }  //0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
    std::cout<<std::endl;

    // output Blob shape
    std::cout << b.shape_string() << std::endl;
    // 1 2 3 4 (24)
}

//void mat2Blob(const cv::Mat& mat, caffe::Blob<float>& blob, std::vector<cv::Mat>* vecMat){
void mat2Blob(const cv::Mat& mat, Blob<float>& blob){

    std::vector<cv::Mat> vecMat;
    blob.Reshape(1, mat.channels(), mat.rows, mat.cols);

    int width = blob.width();
    int height = blob.height();
    float* input_data = blob.mutable_cpu_data();
    for (int i = 0; i < blob.channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        vecMat.push_back(channel);
        input_data += width * height;
    }

    cv::split(mat, vecMat);
}

void blob2Mat(const Blob<float>& blob, cv::Mat& mat){

    for (int c = 0; c < blob.channels(); ++c) {
        for (int h = 0; h < blob.height(); ++h) {
            for (int w = 0; w < blob.width(); ++w) {
                mat.at<cv::Vec3f>(h, w)[c] = blob.data_at(0, c, h, w);
            }
        }
    }

}

void drawMidLine(Blob<float> &blob){
    float* data = blob.mutable_cpu_data();
    // draw black line in the image
    for (int c = 0; c < blob.channels(); ++c) {
        for (int j = 0; j < blob.width(); ++j) {
            *(data + blob.offset(0, c, blob.height() / 2, j)) = 0;
        }
    }
}

void test_blob_2(){
    LOG(INFO) << "now create white image h:240, w240 ";
    // create white image and show it
    cv::Mat img(240, 240, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::imshow("src_white_image", img);

    //convert it to float
    img.convertTo(img, CV_32FC3);

    LOG(INFO) <<"now convert Mat to caffe::Blob";
    // create Blob and convert mat to blob
    Blob<float> blob;
    mat2Blob(img, blob);

    LOG(INFO) <<"now draw line in the mid of  blob";
    drawMidLine(blob);

    LOG(INFO) <<"now convert blob to Mat";
    blob2Mat(blob, img);

    //convert it to CV_8UC3
    img.convertTo(img, CV_8UC3);

    LOG(INFO) <<"now show source image and image with white line";
    // show image
    cv::imshow("image_with_black_line", img);
    cv::waitKey(0);
}

void test_blob_3(){

    const int N = 8;
    vector<int> shape(1, N); // two dimension shape
    Blob<float> blob;
    blob.Reshape(shape);
    cout<<blob.shape_string()<<endl;
    // 8 (8)

    // init blob data with sequential num
    float* pf = blob.mutable_cpu_data();
    std::iota(pf, pf + N, 1.0);

    // output data in blob
    for (int i = 0; i < blob.count(); ++i) {
        cout<<*(pf + i)<<" ";
    }// 1 2 3 4 5 6 7 8
    cout<<endl;

}

void test_blob_filter(){

    Blob<float> blob;
    vector<int> shape{1, 1, 1, 5};
    blob.Reshape(shape);

    FillerParameter param;
    param.set_type("uniform");
    param.set_min(1.0);
    param.set_max(10.0);
    shared_ptr<Filler<float> > filler(GetFiller<float>(param));

    filler->Fill(&blob);

    const float* pf = blob.cpu_data();
    // output data in blob
    for (int i = 0; i < blob.count(); ++i) {
        cout<<*(pf + i)<<" ";
    }// maybe 1.84415 1.00083 4.53987 7.09303 1.54863
    cout<<endl;
}

void test_prototxt(){
    string path = "/home/xy/caffe_analysis/my_caffe/example/lenet.prototxt";

    // read proto txt to proto class
    NetParameter param;
    ReadNetParamsFromTextFileOrDie(path, &param);

    // print proto class
    PrintProto(param);

    // write proto class to txt format
    string path2 = "/home/xy/caffe_analysis/my_caffe/example/lenet2.prototxt";
    WriteProtoToTextFile(param, path2);
}

//    REGISTER_LAYER_CLASS(InnerProduct);
//    REGISTER_LAYER_CLASS(ReLU);
//    REGISTER_LAYER_CLASS(Data);
//    REGISTER_LAYER_CLASS(SoftmaxWithLoss);
//    REGISTER_LAYER_CLASS(Softmax);

    void datum2Blob(const Datum& datum, Blob<float>* blob){


    }


void test_net(){
    Caffe::set_mode(Caffe::CPU);
    string base_dir = "/home/xy/caffe_analysis/my_caffe/example/";
    string net_path = base_dir + "lenet_deploy.prototxt";
    string trained_path = base_dir + "model_iter_20000.caffemodel";

    Net<float> net(net_path, Phase::TEST);
    net.CopyTrainedLayersFrom(trained_path);

    CHECK_EQ(net.num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net.num_outputs(), 1) << "Network should have exactly one output.";

    // read Datum from file
    string datum_path = base_dir + "0.proto";
    Datum datum;
    ReadProtoFromBinaryFileOrDie(datum_path, &datum);

    // Datum convert
    TransformationParameter parameter;
    parameter.set_scale(0.00390625);
    DataTransformer<float> dataTransformer(parameter, Phase::TEST);

    Blob<float>* input_layer = net.input_blobs()[0];
//    input_layer->Reshape(1, 1, 28, 28);//no need
    dataTransformer.Transform(datum, input_layer);

//    net.Reshape();
    net.Forward();
    Blob<float>* output_layer = net.output_blobs()[0];

    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();

    std::vector<float> ret(begin, end);

    for(auto item:ret)cout<<item<<" ";
    cout<<endl;

}

    void test_solver(){

        google::InitGoogleLogging("test_solver");
        FLAGS_alsologtostderr = 1;

        string base_dir = "/home/xy/caffe_analysis/my_caffe/example/";
        string path = base_dir + "mlp_solver.prototxt";
        SGDSolver<float> solver(path);

        // resuming from trained weights
        string trained_weight_path = base_dir + "model_iter_20000.caffemodel";
        solver.net()->CopyTrainedLayersFrom(trained_weight_path);

        // 开始优化
        solver.Solve();
    }

    void test_mnist(){

        LOG(INFO) <<swap_endian(5);

        string base_dir = "/home/xy/caffe-master/data/mnist/";
        string img_path = base_dir + "train-images-idx3-ubyte";
        string label_path = base_dir + "train-labels-idx1-ubyte";

        string db_path = "/home/xy/tmp/lmdb";
//        convert_dataset(img_path.c_str(), label_path.c_str(), db_path.c_str());

        read_mnist_cv(img_path.c_str(), label_path.c_str());
    }


}