//
// Created by root on 9/19/18.
//

#ifndef MY_CAFFE_SGD_SOLVERS_H
#define MY_CAFFE_SGD_SOLVERS_H

#include "solver.h"

namespace caffe{

    template <typename Dtype>
    class SGDSolver : public Solver<Dtype> {
    public:
        explicit SGDSolver(const SolverParameter& param)
                : Solver<Dtype>(param) { PreSolve(); }
        explicit SGDSolver(const string& param_file)
                : Solver<Dtype>(param_file) { PreSolve(); }
        virtual inline const char* type() const { return "SGD"; }

        const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

    protected:
        void PreSolve();
        Dtype GetLearningRate();
        virtual void ApplyUpdate();
        virtual void Normalize(int param_id);
        virtual void Regularize(int param_id);
        virtual void ComputeUpdateValue(int param_id, Dtype rate);
        virtual void ClipGradients();
        virtual void SnapshotSolverState(const string& model_filename);
        virtual void SnapshotSolverStateToBinaryProto(const string& model_filename);
        virtual void SnapshotSolverStateToHDF5(const string& model_filename);
        virtual void RestoreSolverStateFromHDF5(const string& state_file);
        virtual void RestoreSolverStateFromBinaryProto(const string& state_file);
        // history maintains the historical momentum data.
        // update maintains update related data and is not needed in snapshots.
        // temp maintains other information that might be needed in computation
        //   of gradients/updates and is not needed in snapshots
        vector<shared_ptr<Blob<Dtype> > > history_, update_, temp_;

    DISABLE_COPY_AND_ASSIGN(SGDSolver);
    };

}

#endif //MY_CAFFE_SGD_SOLVERS_H
