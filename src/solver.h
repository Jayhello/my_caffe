//
// Created by root on 9/19/18.
//

#ifndef MY_CAFFE_SOLVER_H
#define MY_CAFFE_SOLVER_H

#include <boost/function.hpp>
#include <string>
#include "caffe.pb.h"
#include "net.h"

/*
 * (1)solver_factory的register和create不同类型Solver的机制，
 * (2)通过signal_handler来获取系统信号，并根据用户或默认的设置进行相应的处理，
 * (3)Solver::Solve函数的具体实现的分析，
 * (4)SGDSolver::ApplyUpdate函数的具体实现。前面三个部分都属于基类的，
 * 最后一个是SGDSolver这个子类的，如果用户想要实现自己的Solver类，
 * 也应该类似地去继承基类，并实现自己的ApplyUpdate函数，在代码的末尾通过
 * register宏完成注册，便可以被成功的调用。
 */

namespace caffe{
/*
 * 大概意思就是按Ctrl-C时，会保存当前训练时的模型，
 * 如果还在训练终端不小心被关闭时，可以接着上次继续训练
 */

    namespace SolverAction {
        enum Enum {
            NONE = 0,  // Take no special action.
            STOP = 1,  // Stop training. snapshot_after_train controls whether a
            // snapshot is created.
                    SNAPSHOT = 2  // Take a snapshot, and keep training.
        };
    }

/**
 * @brief Type of a function that returns a Solver Action enumeration.
 学过java的可以理解为回滚操作，比如银行账户钱从一个用户转到另一个账户时，
 中途发生点意外，一个用户钱已经减了，另一个却没有增加，这时需要回滚操作，
 就像这时训练的时候中断了，然后回滚，到上次断点，继续训练。
*/
    typedef boost::function<SolverAction::Enum()> ActionCallback;

    template <typename Dtype>
    class Solver {
    public:
        explicit Solver(const SolverParameter& param,
                        const Solver* root_solver = NULL);
        explicit Solver(const string& param_file, const Solver* root_solver = NULL);

        void Init(const SolverParameter& param);
        void InitTrainNet();
        void InitTestNets();

        // Client of the Solver optionally may call this in order to set the function
        // that the solver uses to see what action it should take (e.g. snapshot or
        // exit training early).
        void SetActionFunction(ActionCallback func);
        SolverAction::Enum GetRequestedAction();

        // The main entry of the solver function. In default, iter will be zero. Pass
        // in a non-zero iter number to resume training for a pre-trained net.
        // 主函数，默认iter为0,非0的iter输入到预训练的网络中。
        virtual void Solve(const char* resume_file = NULL);
        inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
        void Step(int iters);

        //存储函数实现如何存储solver到快照模型中。应该实现RestoreSolverState()函数
        //这个函数是存储来自SolverState缓冲的状态
        void Restore(const char* resume_file);

        void Snapshot();

        //虚析构
        virtual ~Solver() {}

        //返回solver参数
        inline const SolverParameter& param() const { return param_; }

        //返回网络
        inline shared_ptr<Net<Dtype> > net() { return net_; }

        //返回测试网络
        inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
            return test_nets_;
        }
        int iter() { return iter_; }

        // 在迭代中调用特殊的点
        // 嵌套类，外层类的对象与内层类的对象是相互独立的
        // 嵌套类在其外层类中定义了一个类型成员，该类型的访问权限由外层类决定
        class Callback {
        protected:
            virtual void on_start() = 0;
            virtual void on_gradients_ready() = 0;

            template <typename T>
            friend class Solver;
        };

        const vector<Callback*>& callbacks() const { return callbacks_; }
        void add_callback(Callback* value) {
            callbacks_.push_back(value);
        }

        void CheckSnapshotWritePermissions();

        virtual inline const char* type() const { return ""; }

    protected:
        // Make and apply the update value for the current iteration.
        // 纯虚函数，需要派生类实现，生成和应用当前迭代的更新的值
        virtual void ApplyUpdate() = 0;

        string SnapshotFilename(const string extension);
        string SnapshotToBinaryProto();
        string SnapshotToHDF5();

        // The test routine
        // 测试程序
        void TestAll();
        void Test(const int test_net_id = 0);
        virtual void SnapshotSolverState(const string& model_filename) = 0;
        virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
        virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
        void DisplayOutputBlobs(const int net_id);
        void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);

        //Solver参数
        SolverParameter param_;

        int iter_;
        int current_step_;

        // 训练网络，有且只有一个
        shared_ptr<Net<Dtype> > net_;

        // 测试网络可以有多个
        vector<shared_ptr<Net<Dtype> > > test_nets_;

        vector<Callback*> callbacks_;
        vector<Dtype> losses_;
        Dtype smoothed_loss_;

        // 在数据并行中，继续根solver层保持根nets（包含共享的层）
        const Solver* const root_solver_;

        // 通过函数是选择确认按钮来选择保存还是退出快照。
        ActionCallback action_request_function_;

        // True iff a request to stop early was received.
        bool requested_early_exit_;

    DISABLE_COPY_AND_ASSIGN(Solver);
    };

}

#endif //MY_CAFFE_SOLVER_H
