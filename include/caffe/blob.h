//
// Created by root on 7/23/18.
//

#ifndef MY_CAFFE_BLOB_H
#define MY_CAFFE_BLOB_H


#include "syncedmem.h"
#include "proto/caffe.pb.h"


const int kMaxBlobAxes = 32; //在头文件中为它添加 extern 声明,以使其能被多个文件共享

namespace caffe{
    
    void test_print();
    
    template<typename Dtype>
    class Blob{

        public:
            Blob() //构造函数：初始化列表 {空函数体}
                    : data_(), diff_(), count_(0), capacity_(0) {}

        /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
        explicit Blob(const int num, const int channels, const int height,
                      const int width);   //可以通过设置数据维度（N,C,H,W）初始化

        explicit Blob(const vector<int>& shape); //也可以通过传入vector<int>直接传入维数

        /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
        void Reshape(const int num, const int channels, const int height,
                     const int width);

        void Reshape(const vector<int>& shape);
        void Reshape(const BlobShape& shape);
        void ReshapeLike(const Blob& other);

        // 输出blob的形状
        inline string shape_string() const { //
            ostringstream stream;
            for (int i = 0; i < shape_.size(); ++i) {
                stream << shape_[i] << " ";
            }
            stream << "(" << count_ << ")";
            return stream.str();
        }

        //根据索引返回维数，对于维数(N,C,H,W),shape(0)返回N,shape(-1)返回W。
        inline int shape(int index) const {
            return shape_[CanonicalAxisIndex(index)];
        }

        inline const vector<int>& shape() const { return shape_; }

        //返回Blob维度数，对于维数(N,C,H,W)，返回4
        inline int num_axes() const { return shape_.size(); }

        //返回Blob维度数，对于维数(N,C,H,W)，返回N×C×H×W
        inline int count() const { return count_; }

        //对于维数(N,C,H,W)，count(0, 3)返回N×C×H
        inline int count(int start_axis, int end_axis) const {
            CHECK_LE(start_axis, end_axis);
            CHECK_GE(start_axis, 0);
            CHECK_GE(end_axis, 0);
            CHECK_LE(start_axis, num_axes());
            CHECK_LE(end_axis, num_axes());
            int count = 1;
            for (int i = start_axis; i < end_axis; ++i) {
                count *= shape(i);
            }
            return count;
        }

        //对于维数(N,C,H,W)，count(1)返回C×H×W
        inline int count(int start_axis) const {
            return count(start_axis, num_axes());
        }

        inline int CanonicalAxisIndex(int axis_index) const {
            CHECK_GE(axis_index, -num_axes())
                << "axis " << axis_index << " out of range for " << num_axes()
                << "-D Blob with shape " << shape_string();
            CHECK_LT(axis_index, num_axes())
                << "axis " << axis_index << " out of range for " << num_axes()
                << "-D Blob with shape " << shape_string();
            if (axis_index < 0) {
                return axis_index + num_axes();
            }
            return axis_index;
        }

        /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
        inline int num() const { return LegacyShape(0); }
        /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
        inline int channels() const { return LegacyShape(1); }
        /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
        inline int height() const { return LegacyShape(2); }
        /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
        inline int width() const { return LegacyShape(3); }
        inline int LegacyShape(int index) const {
            CHECK_LE(num_axes(), 4)
                << "Cannot use legacy accessors on Blobs with > 4 axes.";
            CHECK_LT(index, 4);
            CHECK_GE(index, -4);
            if (index >= num_axes() || index < -num_axes()) {
                // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
                // indexing) -- this special case simulates the one-padding used to fill
                // extraneous axes of legacy blobs.
                return 1;
            }
            return shape(index);
        }

        //计算物理偏移量，(n,c,h,w)的偏移量为((n∗C+c)∗H+h)∗W+w
        inline int offset(const int n, const int c = 0, const int h = 0,
                          const int w = 0) const {
            CHECK_GE(n, 0);
            CHECK_LE(n, num());
            CHECK_GE(channels(), 0);
            CHECK_LE(c, channels());
            CHECK_GE(height(), 0);
            CHECK_LE(h, height());
            CHECK_GE(width(), 0);
            CHECK_LE(w, width());
            return ((n * channels() + c) * height() + h) * width() + w;
        }

        inline int offset(const vector<int>& indices) const {
            CHECK_LE(indices.size(), num_axes());
            int offset = 0;
            for (int i = 0; i < num_axes(); ++i) {
                offset *= shape(i);
                if (indices.size() > i) {
                    CHECK_GE(indices[i], 0);
                    CHECK_LT(indices[i], shape(i));
                    offset += indices[i];
                }
            }
            return offset;
        }

        void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,   //从source拷贝数据， copy_diff来作为标志区分是拷贝data还是diff。
                      bool reshape = false);

        inline Dtype data_at(const int n, const int c, const int h,
                             const int w) const {
            return cpu_data()[offset(n, c, h, w)];
        }

        inline Dtype diff_at(const int n, const int c, const int h,
                             const int w) const {
            return cpu_diff()[offset(n, c, h, w)];
        }

        inline Dtype data_at(const vector<int>& index) const {
            return cpu_data()[offset(index)];
        }

        inline Dtype diff_at(const vector<int>& index) const {
            return cpu_diff()[offset(index)];
        }

        inline const shared_ptr<SyncedMemory>& data() const {
            CHECK(data_);
            return data_;
        }

        inline const shared_ptr<SyncedMemory>& diff() const {
            CHECK(diff_);
            return diff_;
        }

        /*
            // 假定数据在 CPU 上进行初始化，我们有一个 blob
          const Dtype* foo;
          Dtype* bar;
          foo = blob.gpu_data(); // 数据从 CPU 复制到 GPU
          foo = blob.cpu_data(); // 没有数据复制，两者都有最新的内容
          bar = blob.mutable_gpu_data(); // 没有数据复制
          // ... 一些操作 ...
          bar = blob.mutable_gpu_data(); // 仍在 GPU，没有数据复制
          foo = blob.cpu_data(); // 由于 GPU 修改了数值，数据从 GPU 复制到 CPU
          foo = blob.gpu_data(); //没有数据复制，两者都有最新的内容
          bar = blob.mutable_cpu_data(); // 依旧没有数据复制
          bar = blob.mutable_gpu_data(); //数据从 CPU 复制到 GPU
          bar = blob.mutable_cpu_data(); //数据从 GPU 复制到 CPU

         */

        const Dtype* cpu_data() const;  //数据访问，const方式只读，不允许改写数据
        void set_cpu_data(Dtype* data);
        const int* gpu_shape() const;
        const Dtype* gpu_data() const;
        const Dtype* cpu_diff() const;
        const Dtype* gpu_diff() const;
        Dtype* mutable_cpu_data();      //mutable方式可改写数据（对diff_的访问也是类似的）
        Dtype* mutable_gpu_data();
        Dtype* mutable_cpu_diff();
        Dtype* mutable_gpu_diff();
        void Update();

        //从proto读数据进来，其实就是反序列化
        void FromProto(const BlobProto& proto, bool reshape = true);
        //blob数据保存到proto中
        void ToProto(BlobProto* proto, bool write_diff = false) const;

        /// @brief Compute the sum of absolute values (L1 norm) of the data.
        Dtype asum_data() const;
        /// @brief Compute the sum of absolute values (L1 norm) of the diff.
        Dtype asum_diff() const;
        /// @brief Compute the sum of squares (L2 norm squared) of the data.
        Dtype sumsq_data() const;
        /// @brief Compute the sum of squares (L2 norm squared) of the diff.
        Dtype sumsq_diff() const;

        /// @brief Scale the blob data by a constant factor.
        void scale_data(Dtype scale_factor);
        /// @brief Scale the blob diff by a constant factor.
        void scale_diff(Dtype scale_factor);

        void ShareData(const Blob& other); //Blob& other 赋值给data_

        void ShareDiff(const Blob& other); //Blob& other 赋值给diff_

        bool ShapeEquals(const BlobProto& other);

        protected:
            shared_ptr<SyncedMemory> data_; //存储前向传递数据
            shared_ptr<SyncedMemory> diff_; //存储反向传递梯度
            shared_ptr<SyncedMemory> shape_data_;
            vector<int> shape_;  //参数维度
            int count_; //Blob存储的元素个数（shape_所有元素乘积）
            int capacity_;//当前Blob的元素个数（控制动态分配）

            DISABLE_COPY_AND_ASSIGN(Blob);
    };
}


#endif //MY_CAFFE_BLOB_H
