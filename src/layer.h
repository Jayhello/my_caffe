//
// Created by root on 7/24/18.
//

#ifndef MY_CAFFE_LAYER_H
#define MY_CAFFE_LAYER_H

#include "common.h"
#include "blob.h"
#include "math_functions.h"
#include "caffe.pb.h"

namespace boost{ class mutex; }


namespace caffe{

    template<typename Dtype>
    class Layer{

    protected:
        //protobuf文件中存储的layer参数,从protocal buffers格式的网络结构说明文件中读取
        //protected类成员，构造函数中初始化
        LayerParameter layer_param_;

        //层状态，参与网络的训练还是测试
        Phase phase_;

        // 可学习参数层权值和偏置参数，使用向量是因为权值参数和偏置是分开保存在两个blob中的
        // 在基类layer中初始化(只是在描述文件定义了的情况下)
        vector<shared_ptr<Blob<Dtype> > > blobs_;

        // 标志每个可学习参数blob是否需要计算反向传递的梯度值
        vector<bool> param_propagate_down_;

        // 非LossLayer为零，LossLayer中表示每个top blob计算的loss的权重
        vector<Dtype> loss_;

    private:
        /** Whether this layer is actually shared by other nets*/
        bool is_shared_;

        // 若该layer被shared，则需要这个mutex序列保持forward过程的正常运行
        shared_ptr<boost::mutex> forward_mutex_;

        DISABLE_COPY_AND_ASSIGN(Layer);

    public:
        explicit Layer(const LayerParameter& param)
        :layer_param_(param), is_shared_(false){
            phase_ = param.phase();

        }

        // 虚析构
        virtual ~Layer() {}

    public:

        /**
         * 从 bottom 层中接收数据，进行计算后将输出送入到 top 层中;
         * 这两个函数非虚函数，它们内部会调用如下虚函数(Forward_cpu and (optionally) Forward_gpu)完成数据前向传递和误差反向传播，
         * 根据执行环境的不同每个子类Layer必须重写CPU和GPU版本
         *
         */
        inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        /*
         * 给定相对于 top 层输出的梯度，计算其相对于输入的梯度，并传递到 bottom
         * 层。一个有参数的 layer 需要计算相对于各个参数的梯度值并存储在内部。
         */
        inline void Backward(const vector<Blob<Dtype>*>& top,
                             const vector<bool>& propagate_down,
                             const vector<Blob<Dtype>*>& bottom);

    protected:
        /*
         * 纯虚函数，子类必须实现，使用cpu经行前向计算
         */
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) = 0;

        /*
         * 使用gpu经行前向计算, 如果gpu没有实现则使用默认的CPU版本
         */
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {
            // LOG(WARNING) << "Using CPU code as backup.";
            return Forward_cpu(bottom, top);
        }

        /*
         * 纯虚函数，派生类必须实现
         */
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down,
                                  const vector<Blob<Dtype>*>& bottom) = 0;

        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down,
                                  const vector<Blob<Dtype>*>& bottom) {
            // LOG(WARNING) << "Using CPU code as backup.";
            Backward_cpu(top, propagate_down, bottom);
        }

    public:
        /*
         * 根据bottom blob的形状和layer_param_计算top blob的形状并为其分配存储空间
         * 每个子类Layer必须重写的Reshape函数，完成top blob形状的设置并为其分配存储空间
         */
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top) = 0;

        // 给定index返回相应的scalar loss
        inline Dtype loss(const int top_index) const {
            return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
        }

        virtual void ToProto(LayerParameter* param, bool write_diff = false);

        // layer 初始化设置
        //在模型初始化时重置 layers 及其相互之间的连接 ;
        void SetUp(const vector<Blob<Dtype>*>& bottom,
                   const vector<Blob<Dtype>*>& top) {
            InitMutex();
            CheckBlobCounts(bottom, top);
            LayerSetUp(bottom, top);
            Reshape(bottom, top);
            SetLossWeights(top);
        }

    public:
        virtual inline const char* type() const { return ""; }

        virtual inline int ExactNumBottomBlobs() const { return -1; }
        virtual inline int MinBottomBlobs() const { return -1; }
        virtual inline int MaxBottomBlobs() const { return -1; }
        virtual inline int ExactNumTopBlobs() const { return -1; }
        virtual inline int MinTopBlobs() const { return -1; }
        virtual inline int MaxTopBlobs() const { return -1; }
        virtual inline bool EqualNumBottomTopBlobs() const { return false; }

        // 检查输出输出的blobs的个数是否在给定范围内
        virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
            if (ExactNumBottomBlobs() >= 0) {
                CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
                    << type() << " Layer takes " << ExactNumBottomBlobs()
                    << " bottom blob(s) as input.";
            }
            if (MinBottomBlobs() >= 0) {
                CHECK_LE(MinBottomBlobs(), bottom.size())
                    << type() << " Layer takes at least " << MinBottomBlobs()
                    << " bottom blob(s) as input.";
            }
            if (MaxBottomBlobs() >= 0) {
                CHECK_GE(MaxBottomBlobs(), bottom.size())
                    << type() << " Layer takes at most " << MaxBottomBlobs()
                    << " bottom blob(s) as input.";
            }
            if (ExactNumTopBlobs() >= 0) {
                CHECK_EQ(ExactNumTopBlobs(), top.size())
                    << type() << " Layer produces " << ExactNumTopBlobs()
                    << " top blob(s) as output.";
            }
            if (MinTopBlobs() >= 0) {
                CHECK_LE(MinTopBlobs(), top.size())
                    << type() << " Layer produces at least " << MinTopBlobs()
                    << " top blob(s) as output.";
            }
            if (MaxTopBlobs() >= 0) {
                CHECK_GE(MaxTopBlobs(), top.size())
                    << type() << " Layer produces at most " << MaxTopBlobs()
                    << " top blob(s) as output.";
            }
            if (EqualNumBottomTopBlobs()) {
                CHECK_EQ(bottom.size(), top.size())
                    << type() << " Layer produces one top blob as output for each "
                    << "bottom blob input.";
            }
        }

        /*
         * 定制初始化，每个子类layer必须实现此虚函数
         * 此方法执行一次定制化的层初始化，包括从layer_param_读入并处理相关的层权值和偏置参数，
         * 调用Reshape函数申请top blob的存储空间,由派生类重写
         */
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top) {}

        inline void SetLossWeights(const vector<Blob<Dtype>*>& top) {
            const int num_loss_weights = layer_param_.loss_weight_size();
            if (num_loss_weights) {
                CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
                            "unspecified or specified once per top blob.";
                for (int top_id = 0; top_id < top.size(); ++top_id) {
                    const Dtype loss_weight = layer_param_.loss_weight(top_id);
                    if (loss_weight == Dtype(0)) { continue; }
                    this->set_loss(top_id, loss_weight);
                    const int count = top[top_id]->count();
                    Dtype* loss_multiplier = top[top_id]->mutable_cpu_diff();
                    caffe_set(count, loss_weight, loss_multiplier);
                }
            }
        }

        inline void set_loss(const int top_index, const Dtype value) {
            if (loss_.size() <= top_index) {
                loss_.resize(top_index + 1, Dtype(0));
            }
            loss_[top_index] = value;
        }

        virtual inline bool ShareInParallel() const { return false; }

        inline void SetShared(bool is_shared) {
            CHECK(ShareInParallel() || !is_shared)
            << type() << "Layer does not support sharing.";
            is_shared_ = is_shared;
        }

        /**
         * @brief Sets whether the layer should compute gradients w.r.t. a
         *        parameter at a particular index given by param_id.
         *        设置是否对某个学习参数blob计算梯度
         */
        inline void set_param_propagate_down(const int param_id, const bool value) {
            if (param_propagate_down_.size() <= param_id) {
                param_propagate_down_.resize(param_id + 1, true);
            }
            param_propagate_down_[param_id] = value;
        }

        inline bool param_propagate_down(const int param_id) {
            return (param_propagate_down_.size() > param_id) ?
                   param_propagate_down_[param_id] : false;
        }

        virtual inline bool AllowForceBackward(const int bottom_index) const {
            return true;
        }

        const LayerParameter& layer_param() const { return layer_param_; }

        /**
         * @brief Returns the vector of learnable parameter blobs.
         * 返回可学习的参数blobs
         */
        vector<shared_ptr<Blob<Dtype> > >& blobs() {
            return blobs_;
        }

        /**
         * @brief Return whether "anonymous" top blobs are created automatically
         *        by the layer.
         *
         * If this method returns true, Net::Init will create enough "anonymous" top
         * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
         * MinTopBlobs().
         */
        virtual inline bool AutoTopBlobs() const { return false; }

    private:

        /** Initialize forward_mutex_ */
        void InitMutex();
        /** Lock forward_mutex_ if this layer is shared */
        void Lock();
        /** Unlock forward_mutex_ if this layer is shared */
        void Unlock();

    public:
        inline bool IsShared() const { return is_shared_; }

    };

    // 前向传播和反向传播接口。 每个Layer的派生类都应该实现Forward_cpu()
    template <typename Dtype>
    inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
        // Lock during forward to ensure sequential forward
        Lock();
        Dtype loss = 0;
        Reshape(bottom, top);
        switch (Caffe::mode()) {
            case Caffe::CPU:
                Forward_cpu(bottom, top);

                // 计算loss
                for (int top_id = 0; top_id < top.size(); ++top_id) {
                    if (!this->loss(top_id)) { continue; }
                    const int count = top[top_id]->count();
                    const Dtype* data = top[top_id]->cpu_data();
                    const Dtype* loss_weights = top[top_id]->cpu_diff();
                    loss += caffe_cpu_dot(count, data, loss_weights);
                }
                break;
            case Caffe::GPU:
                Forward_gpu(bottom, top);
    #ifndef CPU_ONLY
                // gpu realize, omitted
    #endif
                break;
            default:
                LOG(FATAL) << "Unknown caffe mode.";
        }
        Unlock();
        return loss;
    }

    template <typename Dtype>
    inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom) {
        switch (Caffe::mode()) {
            case Caffe::CPU:
                Backward_cpu(top, propagate_down, bottom);
                break;
            case Caffe::GPU:
                Backward_gpu(top, propagate_down, bottom);
                break;
            default:
                LOG(FATAL) << "Unknown caffe mode.";
        }
    }

    //Layer的序列化函数,将layer的层说明参数layer_param_，
    //层权值和偏置参数blobs_复制到LayerParameter对象，便于写到磁盘
    template <typename Dtype>
    void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
        param->Clear();
        param->CopyFrom(layer_param_);
        param->clear_blobs();
        // 复制层权值和偏置参数blobs_
        for (int i = 0; i < blobs_.size(); ++i) {
            blobs_[i]->ToProto(param->add_blobs(), write_diff);
        }
    }

}


#endif //MY_CAFFE_LAYER_H
