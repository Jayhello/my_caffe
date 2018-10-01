project(caffe_lib)
set(CMAKE_CXX_STANDARD 11)

#include_directories(${INC_DIR})

#set(SRC blob.cpp caffe.cpp syncedmem.cpp math_functions.cpp)
#file(GLOB SRC_LIST ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/util/.*cpp)
aux_source_directory(. SRC)

message("caffe_lib use files: ${SRC}")

add_library(caffe_lib SHARED ${SRC})
set_target_properties(
        caffe_lib
        PROPERTIES
#        ARCHIVE_OUTPUT_DIRECTORY "${LIB_OUTPUT_PATH}"
        ARCHIVE_OUTPUT_DIRECTORY "/home/xy/caffe_analysis/my_caffe/lib/"
#        LIBRARY_OUTPUT_DIRECTORY "${LIB_OUTPUT_PATH}"
        LIBRARY_OUTPUT_DIRECTORY "/home/xy/caffe_analysis/my_caffe/lib/"
        )

target_link_libraries(
        caffe_lib
        ${GLOG_LIBRARIES}
        ${PROTOBUF_LIBRARIES}
        ${OpenBLAS_LIB}
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        ${LMDB_LIBRARIES}
)