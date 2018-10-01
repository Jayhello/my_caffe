cmake_minimum_required(VERSION 3.5)
project(my_caffe)

set(CMAKE_CXX_STANDARD 11)

#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

add_definitions(-DCPU_ONLY=1)
add_definitions(-DUSE_LMDB=1)

# set variable for sub directory to use
set(INC_DIR "${CMAKE_SOURCE_DIR}/include")
set(LIB_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/lib")

# dependence lib
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(GLog REQUIRED)
find_package(ProtoBuf REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(Opencv REQUIRED)
find_package(LMDB REQUIRED)

find_package(Boost COMPONENTS system thread REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
message("find boost lib: ${Boost_LIBRARIES}")


add_subdirectory(src)

include_directories(${CMAKE_SOURCE_DIR}/src)

set(SOURCE_FILES main.cpp test_blob.h test_blob.cpp convert_mnist_data.cpp classification.cpp)
link_directories(${LIB_OUTPUT_PATH})
add_executable(my_caffe_test ${SOURCE_FILES})

target_link_libraries(my_caffe_test
        /home/xy/caffe_analysis/my_caffe/lib/libcaffe_lib.so
        ${GLOG_LIBRARIES}
        ${PROTOBUF_LIBRARIES}
        ${OpenBLAS_LIB}
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        ${LMDB_LIBRARIES}
        -lcaffe_lib
        )