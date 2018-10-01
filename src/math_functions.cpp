//
// Created by root on 7/23/18.
//

#include <cblas.h>
#include "math_functions.h"
#include "common.h"
#include <boost/random.hpp>
#include <boost/math/special_functions/next.hpp>
#include "rng.h"
#include "mkl_alternate.h"
#include "device_alternate.h"


namespace caffe{

    /*
 * 功能：X = alpha*X
 * N： X中element的个数
 */
    template <>
    void caffe_scal<float>(const int N, const float alpha, float *X) {
        cblas_sscal(N, alpha, X, 1);
    }

    template <>
    void caffe_scal<double>(const int N, const double alpha, double *X) {
        cblas_dscal(N, alpha, X, 1);
    }

    // TODO, if not define int/unsigned in template then
    // undefined reference to `void caffe::caffe_scal<int>
    template <>
    void caffe_scal<int>(const int N, const int alpha, int *X) {
        cblas_csscal(N, alpha, X, 1);
    }

    template <>
    void caffe_scal<unsigned int>(const int N, const unsigned int alpha, unsigned int *X) {
        cblas_csscal(N, alpha, X, 1);
    }


    /*
     * 功能： Y=alpha*X+Y
     * N：为X和Y中element的个数
     */
    template <>
    void caffe_axpy<float>(const int N, const float alpha, const float* X,
                           float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

    template <>
    void caffe_axpy<double>(const int N, const double alpha, const double* X,
                            double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

    template <>
    float caffe_cpu_asum<float>(const int n, const float* x) {
        return cblas_sasum(n, x, 1);
    }

    template <>
    double caffe_cpu_asum<double>(const int n, const double* x) {
        return cblas_dasum(n, x, 1);
    }

    /*
     * 功能： 返回 vector X 和 vector Y 的内积。
     * incx， incy ： 步长，即每隔incx 或 incy 个element 进行操作。
     */
    template <>
    float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
                                       const float* y, const int incy) {
        return cblas_sdot(n, x, incx, y, incy);
    }

    template <>
    double caffe_cpu_strided_dot<double>(const int n, const double* x,
                                         const int incx, const double* y, const int incy) {
        return cblas_ddot(n, x, incx, y, incy);
    }


    /*
 * 功能： 返回 vector X 和 vector Y 的内积。
 */
    template <typename Dtype>
    Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
        return caffe_cpu_strided_dot(n, x, 1, y, 1);
    }

    /*
     * or undefined reference
     */
    template
    float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

    template
    double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

    template <typename Dtype>
    void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
        if (alpha == 0) {
            memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
            return;
        }
        for (int i = 0; i < N; ++i) {
            Y[i] = alpha;
        }
    }

    template <typename Dtype>
    void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
        if (X != Y) {
            if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
                //TODO UNDO
                // NOLINT_NEXT_LINE(caffe/alt_fn)
                //CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
                NO_GPU;
#endif
            } else {
                memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
            }
        }
    }

    template void caffe_copy<int>(const int N, const int* X, int* Y);
    template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
                                           unsigned int* Y);
    template void caffe_copy<float>(const int N, const float* X, float* Y);
    template void caffe_copy<double>(const int N, const double* X, double* Y);

    /*
     * template
     * 功能：用常数 alpha 对 Y 进行初始化
     * 函数 void *memset(void *buffer, char c, unsigned count) 一般为新申请的内存做初始化，
     * 功能是将buffer所指向内存中的每个字节的内容全部设置为c指定的ASCII值, count为块的大小
     *
     */
    template void caffe_set<int>(const int N, const int alpha, int* Y);
    template void caffe_set<float>(const int N, const float alpha, float* Y);
    template void caffe_set<double>(const int N, const double alpha, double* Y);

    template <typename Dtype>
    Dtype caffe_nextafter(const Dtype b) {
        return boost::math::nextafter<Dtype>(
                b, std::numeric_limits<Dtype>::max());
    }

    template
    float caffe_nextafter(const float b);

    template
    double caffe_nextafter(const double b);


    template <typename Dtype>
    void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
        CHECK_GE(n, 0);
        CHECK(r);
        CHECK_LE(a, b);
        boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
        boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
                                                variate_generator(caffe_rng(), random_distribution);
        for (int i = 0; i < n; ++i) {
            r[i] = variate_generator();
        }
    }

    template
    void caffe_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r);

    template
    void caffe_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r);

    template <typename Dtype>
    void caffe_rng_gaussian(const int n, const Dtype a,
                            const Dtype sigma, Dtype* r) {
        CHECK_GE(n, 0);
        CHECK(r);
        CHECK_GT(sigma, 0);
        boost::normal_distribution<Dtype> random_distribution(a, sigma);
        boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
                variate_generator(caffe_rng(), random_distribution);
        for (int i = 0; i < n; ++i) {
            r[i] = variate_generator();
        }
    }

    template
    void caffe_rng_gaussian<float>(const int n, const float mu,
                                   const float sigma, float* r);

    template
    void caffe_rng_gaussian<double>(const int n, const double mu,
                                    const double sigma, double* r);

    template <typename Dtype>
    void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
        CHECK_GE(n, 0);
        CHECK(r);
        CHECK_GE(p, 0);
        CHECK_LE(p, 1);
        boost::bernoulli_distribution<Dtype> random_distribution(p);
        boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
                variate_generator(caffe_rng(), random_distribution);
        for (int i = 0; i < n; ++i) {
            r[i] = variate_generator();
        }
    }

    template
    void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

    template
    void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

    template <typename Dtype>
    void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
        CHECK_GE(n, 0);
        CHECK(r);
        CHECK_GE(p, 0);
        CHECK_LE(p, 1);
        boost::bernoulli_distribution<Dtype> random_distribution(p);
        boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
                variate_generator(caffe_rng(), random_distribution);
        for (int i = 0; i < n; ++i) {
            r[i] = static_cast<unsigned int>(variate_generator());
        }
    }

    template
    void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

    template
    void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

    /*
        功能： C=alpha*A*B+beta*C
        A,B,C 是输入矩阵（一维数组格式）

        CblasRowMajor :数据是行主序的（二维数据也是用一维数组储存的）
        TransA, TransB：是否要对A和B做转置操作（CblasTrans CblasNoTrans）
        M： A、C 的行数
        N： B、C 的列数
        K： A 的列数， B 的行数
        lda ： A的列数（不做转置）行数（做转置）
        ldb： B的列数（不做转置）行数（做转置）
        A:[M,K]
        B:[K,N]
    */
    template<>
    void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                               const float alpha, const float* A, const float* B, const float beta,
                               float* C) {
        int lda = (TransA == CblasNoTrans) ? K : M;
        int ldb = (TransB == CblasNoTrans) ? N : K;
        cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
                    ldb, beta, C, N);
    }

    template<>
    void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
                                const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                                const double alpha, const double* A, const double* B, const double beta,
                                double* C) {
        int lda = (TransA == CblasNoTrans) ? K : M;
        int ldb = (TransB == CblasNoTrans) ? N : K;
        cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
                    ldb, beta, C, N);
    }

    /*
        功能： y=alpha*A*x+beta*y
        其中X和Y是向量，A 是矩阵
        M：A 的行数
        N：A 的列数
        cblas_sgemv 中的 参数1 表示对X和Y的每个元素都进行操作
    */
    template <>
    void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
                               const int N, const float alpha, const float* A, const float* x,
                               const float beta, float* y) {
        cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
    }

    template <>
    void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
                                const int N, const double alpha, const double* A, const double* x,
                                const double beta, double* y) {
        cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
    }

    template <>
    void caffe_add<float>(const int n, const float* a, const float* b,
                          float* y) {
        vsAdd(n, a, b, y);
    }

    template <>
    void caffe_add<double>(const int n, const double* a, const double* b,
                           double* y) {
        vdAdd(n, a, b, y);
    }

    template <>
    void caffe_sub<float>(const int n, const float* a, const float* b,
                          float* y) {
        vsSub(n, a, b, y);
    }

    template <>
    void caffe_sub<double>(const int n, const double* a, const double* b,
                           double* y) {
        vdSub(n, a, b, y);
    }

    template <>
    void caffe_mul<float>(const int n, const float* a, const float* b,
                          float* y) {
        vsMul(n, a, b, y);
    }

    template <>
    void caffe_mul<double>(const int n, const double* a, const double* b,
                           double* y) {
        vdMul(n, a, b, y);
    }

    template <>
    void caffe_div<float>(const int n, const float* a, const float* b,
                          float* y) {
        vsDiv(n, a, b, y);
    }

    template <>
    void caffe_div<double>(const int n, const double* a, const double* b,
                           double* y) {
        vdDiv(n, a, b, y);
    }

    template <>
    void caffe_exp<float>(const int n, const float* a, float* y) {
        vsExp(n, a, y);
    }

    template <>
    void caffe_exp<double>(const int n, const double* a, double* y) {
        vdExp(n, a, y);
    }

    template <>
    void caffe_log<float>(const int n, const float* a, float* y) {
        vsLn(n, a, y);
    }

    template <>
    void caffe_log<double>(const int n, const double* a, double* y) {
        vdLn(n, a, y);
    }

    template <>
    void caffe_abs<float>(const int n, const float* a, float* y) {
        vsAbs(n, a, y);
    }

    template <>
    void caffe_abs<double>(const int n, const double* a, double* y) {
        vdAbs(n, a, y);
    }

    template <>
    void caffe_add_scalar(const int N, const float alpha, float* Y) {
        for (int i = 0; i < N; ++i) {
            Y[i] += alpha;
        }
    }

    template <>
    void caffe_add_scalar(const int N, const double alpha, double* Y) {
        for (int i = 0; i < N; ++i) {
            Y[i] += alpha;
        }
    }

    template <>
    void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                                const float beta, float* Y) {
        cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
    }

    template <>
    void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                                 const double beta, double* Y) {
        cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
    }

    unsigned int caffe_rng_rand() {
        return (*caffe_rng())();
    }

}
