//
// Created by root on 9/13/18.
//

#include <string>
#include "io.h"
#include "upgrade_proto.h"

namespace caffe{


    void ReadNetParamsFromTextFileOrDie(const string& param_file,
                                        NetParameter* param) {
        CHECK(ReadProtoFromTextFile(param_file, param))
                << "Failed to parse NetParameter file: " << param_file;

        // TODO seems just for legacy usage
        // UpgradeNetAsNeeded(param_file, param);
    }

    void ReadSolverParamsFromTextFileOrDie(const string& param_file,
                                           SolverParameter* param) {
        CHECK(ReadProtoFromTextFile(param_file, param))
        << "Failed to parse SolverParameter file: " << param_file;
//        UpgradeSolverAsNeeded(param_file, param);
    }

    void ReadNetParamsFromBinaryFileOrDie(const string& param_file,
                                          NetParameter* param) {
        CHECK(ReadProtoFromBinaryFile(param_file, param))
        << "Failed to parse NetParameter file: " << param_file;
//        UpgradeNetAsNeeded(param_file, param);
    }

}
