//
// Created by root on 9/13/18.
//

#ifndef MY_CAFFE_UPGRADE_PROTO_H
#define MY_CAFFE_UPGRADE_PROTO_H

#include <string>
#include "caffe.pb.h"

namespace caffe{
// Read parameters from a file into a NetParameter proto message.
    void ReadNetParamsFromTextFileOrDie(const string& param_file,
                                        NetParameter* param);

/*
 * 这里调用了先后调用了两个函数，首先是ReadProtoFromTextFile，这个函数的作用是从param_file
 * 这个路径去读取solver的定义，并将文件中的内容解析存到param这个指针指向的对象;
 * 然后UpgradeSolverAsNeeded完成了新老版本caffe.proto的兼容处理
 */
    void ReadSolverParamsFromTextFileOrDie(const string& param_file,
                                           SolverParameter* param);

    void ReadNetParamsFromBinaryFileOrDie(const string& param_file,
                                          NetParameter* param);

}

#endif //MY_CAFFE_UPGRADE_PROTO_H
