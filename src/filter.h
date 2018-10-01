//
// Created by root on 9/11/18.
//

#ifndef MY_CAFFE_FILLER_H
#define MY_CAFFE_FILLER_H

#include "caffe.pb.h"
#include "blob.h"
#include "common.h"
#include "rng.h"
#include "math_functions.h"


namespace caffe{
    //在网络初始化时，根据layer的定义进行初始参数的填充。
    template <typename Dtype>
    class Filler {
    public:
        explicit Filler(const FillerParameter& param) : filler_param_(param) {}
        virtual ~Filler() {}
        virtual void Fill(Blob<Dtype>* blob) = 0;
    protected:
        FillerParameter filler_param_;
    };  // class Filler

    template <typename Dtype>
    class ConstantFiller : public Filler<Dtype> {
    public:
        explicit ConstantFiller(const FillerParameter& param)
                : Filler<Dtype>(param) {}
        virtual void Fill(Blob<Dtype>* blob) {
            Dtype* data = blob->mutable_cpu_data();
            const int count = blob->count();
            const Dtype value = this->filler_param_.value();
            CHECK(count);
            for (int i = 0; i < count; ++i) {
                data[i] = value;
            }
            CHECK_EQ(this->filler_param_.sparse(), -1)
                << "Sparsity not supported by this Filler.";
        }
    };

    /// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$.
    template <typename Dtype>
    class UniformFiller : public Filler<Dtype> {
    public:
        explicit UniformFiller(const FillerParameter& param)
                : Filler<Dtype>(param) {}
        virtual void Fill(Blob<Dtype>* blob) {
            CHECK(blob->count());
            caffe_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_.min()),
                                     Dtype(this->filler_param_.max()), blob->mutable_cpu_data());
            CHECK_EQ(this->filler_param_.sparse(), -1)
                << "Sparsity not supported by this Filler.";
        }
    };

/// @brief Fills a Blob with Gaussian-distributed values @f$ x = a @f$.
    template <typename Dtype>
    class GaussianFiller : public Filler<Dtype> {
    public:
        explicit GaussianFiller(const FillerParameter& param)
                : Filler<Dtype>(param) {}
        virtual void Fill(Blob<Dtype>* blob) {
            Dtype* data = blob->mutable_cpu_data();
            CHECK(blob->count());
            caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
                                      Dtype(this->filler_param_.std()), blob->mutable_cpu_data());
            int sparse = this->filler_param_.sparse();
            CHECK_GE(sparse, -1);
            if (sparse >= 0) {
                // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
                // These have num == channels == 1; width is number of inputs; height is
                // number of outputs.  The 'sparse' variable specifies the mean number
                // of non-zero input weights for a given output.
                CHECK_GE(blob->num_axes(), 1);
                const int num_outputs = blob->shape(0);
                Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
                rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
                int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
                caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
                for (int i = 0; i < blob->count(); ++i) {
                    data[i] *= mask[i];
                }
            }
        }

    protected:
        shared_ptr<SyncedMemory> rand_vec_;
    };

    template <typename Dtype>
    class XavierFiller : public Filler<Dtype> {
    public:
        explicit XavierFiller(const FillerParameter& param)
                : Filler<Dtype>(param) {}
        virtual void Fill(Blob<Dtype>* blob) {
            CHECK(blob->count());
            int fan_in = blob->count() / blob->num();
            int fan_out = blob->count() / blob->channels();
            Dtype n = fan_in;  // default to fan_in
            if (this->filler_param_.variance_norm() ==
                FillerParameter_VarianceNorm_AVERAGE) {
                n = (fan_in + fan_out) / Dtype(2);
            } else if (this->filler_param_.variance_norm() ==
                       FillerParameter_VarianceNorm_FAN_OUT) {
                n = fan_out;
            }
            Dtype scale = sqrt(Dtype(3) / n);
            caffe_rng_uniform<Dtype>(blob->count(), -scale, scale,
                                     blob->mutable_cpu_data());
            CHECK_EQ(this->filler_param_.sparse(), -1)
                << "Sparsity not supported by this Filler.";
        }
    };

    template <typename Dtype>
    Filler<Dtype>* GetFiller(const FillerParameter& param) {
        const std::string& type = param.type();
        if (type == "constant") {
            return new ConstantFiller<Dtype>(param);
        } else if (type == "gaussian") {
            return new GaussianFiller<Dtype>(param);
        } else if (type == "uniform") {
            return new UniformFiller<Dtype>(param);
        } else if (type == "xavier") {
            return new XavierFiller<Dtype>(param);
        } else {
            CHECK(false) << "Unknown filler name: " << param.type();
        }
        return (Filler<Dtype>*)(NULL);
    }

}

#endif //MY_CAFFE_FILLER_H
