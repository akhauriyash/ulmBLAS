#include BLAS_HEADER
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/level3/gemm.h>

extern "C" {

void
ULMBLAS(dgemm)(enum CBLAS_TRANSPOSE  transA,
               enum CBLAS_TRANSPOSE  transB,
               int                   m,
               int                   n,
               int                   k,
               double                alpha,
               const double          *A,
               int                   ldA,
               const double          *B,
               int                   ldB,
               double                beta,
               double                *C,
               int                   ldC)
{
//
//  Start the operations.
//
    if (transB==CblasNoTrans || transB==AtlasConj) {
        if (transA==CblasNoTrans || transA==AtlasConj) {
//
//          Form  C := alpha*A*B + beta*C.
//
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          A, 1, ldA,
                          B, 1, ldB,
                          beta,
                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B + beta*C
//
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          A, ldA, 1,
                          B, 1, ldB,
                          beta,
                          C, 1, ldC);
        }
    } else {
        if (transA==CblasNoTrans || transA==AtlasConj) {
//
//          Form  C := alpha*A*B**T + beta*C
//
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          A, 1, ldA,
                          B, ldB, 1,
                          beta,
                          C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B**T + beta*C
//
            ulmBLAS::gemm(m, n, k,
                          alpha,
                          A, ldA, 1,
                          B, ldB, 1,
                          beta,
                          C, 1, ldC);
        }
    }
}

void
CBLAS(dgemm)(enum CBLAS_ORDER      order,
             enum CBLAS_TRANSPOSE  transA,
             enum CBLAS_TRANSPOSE  transB,
             int                   m,
             int                   n,
             int                   k,
             double                alpha,
             const double          *A,
             int                   ldA,
             const double          *B,
             int                   ldB,
             double                beta,
             double                *C,
             int                   ldC)
{
//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA;
    int numRowsB;

    if (order==CblasColMajor) {
        numRowsA = (transA==CblasNoTrans || transA==AtlasConj) ? m : k;
        numRowsB = (transB==CblasNoTrans || transB==AtlasConj) ? k : n;
    } else {
        numRowsB = (transB==CblasNoTrans || transB==AtlasConj) ? n : k;
        numRowsA = (transA==CblasNoTrans || transA==AtlasConj) ? k : m;
    }


//
//  Test the input parameters
//
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (transA!=CblasNoTrans && transA!=CblasTrans
     && transA!=AtlasConj && transA!=CblasConjTrans)
    {
        info = 2;
    } else if (transB!=CblasNoTrans && transB!=CblasTrans
            && transB!=AtlasConj && transB!=CblasConjTrans)
    {
        info = 3;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 4;
            } else if (n<0) {
                info = 5;
            } else if (k<0) {
                info = 6;
            } else if (ldA<std::max(1,numRowsA)) {
                info = 9;
            } else if (ldB<std::max(1,numRowsB)) {
                info = 11;
            } else if (ldC<std::max(1,m)) {
                info = 14;
            }
        } else {
            if (n<0) {
                info = 4;
            } else if (m<0) {
                info = 5;
            } else if (k<0) {
                info = 6;
            } else if (ldB<std::max(1,numRowsB)) {
                info = 9;
            } else if (ldA<std::max(1,numRowsA)) {
                info = 11;
            } else if (ldC<std::max(1,n)) {
                info = 14;
            }
         }
    }

    if (info!=0) {
        CBLAS(xerbla)(info, "cblas_dgemm", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(dgemm)(transA, transB, m, n, k, alpha,
                       A, ldA,
                       B, ldB,
                       beta,
                       C, ldC);
    } else {
        ULMBLAS(dgemm)(transB, transA, n, m, k, alpha,
                       B, ldB,
                       A, ldA,
                       beta,
                       C, ldC);
    }
}

} // extern "C"