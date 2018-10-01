//
// Created by root on 9/18/18.
//

#include "db.h"
#include "db_lmdb.h"
#include <string>


namespace caffe {
    namespace db {

        DB *GetDB(DataParameter::DB backend) {
            switch (backend) {
                
     case DataParameter_DB_LMDB:
    return new LMDB();

                default:
                    LOG(FATAL) << "Unknown database backend";
                    return NULL;
            }
        }

        
        
        
        DB *GetDB(const string &backend) {

//#ifdef USE_LEVELDB // TODO
//            if (backend == "leveldb") {
//    return new LevelDB();
//  }
//#endif  // USE_LEVELDB
            
            if (backend == "lmdb") {
                return new LMDB();
            }
            LOG(FATAL) << "Unknown database backend";
            return NULL;
        }


    }// namespace db
}// namespace caffe