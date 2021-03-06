#ifndef ULMBLAS_LEVEL2_HBUMV_TCC
#define ULMBLAS_LEVEL2_HBUMV_TCC 1

#include <ulmblas/auxiliary/real.h>
#include <ulmblas/level1extensions/dotaxpy.h>
#include <ulmblas/level1/scal.h>
#include <ulmblas/level2/hbumv.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
hbumv(IndexType    n,
      IndexType    k,
      const Alpha  &alpha,
      bool         conjA,
      const TA     *A,
      IndexType    ldA,
      const TX     *x,
      IndexType    incX,
      const Beta   &beta,
      TY           *y,
      IndexType    incY)
{
    typedef decltype(Alpha(0)*TA(0)*TX(0)+Beta(0)*TY(0))  T;

    if (n==0 || (alpha==Alpha(0) && beta==Beta(1))) {
        return;
    }

    scal(n, beta, y, incY);

    if (alpha==Alpha(0)) {
        return;
    }

    for (IndexType j=0; j<n; ++j) {
        IndexType i0  = std::max(IndexType(0), k-j);
        IndexType i1  = k - std::max(IndexType(0), (j+1)-n);
        IndexType len = std::max(IndexType(0), i1-i0+1);

        IndexType i0XY = std::max(IndexType(0), j-k);
        IndexType i1XY = i0XY + len - 1;
        T         rho;

        dotaxpy(len-1, conjA, !conjA, false,
                alpha*x[j*incX],
                &A[i0], IndexType(1),
                &x[i0XY*incX], incX,
                &y[i0XY*incY], incY,
                rho);
        y[i1XY*incY] += alpha*(rho+real(A[i1])*x[i1XY*incX]);
        A += ldA;
    }
}

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
hbumv(IndexType    n,
      IndexType    k,
      const Alpha  &alpha,
      const TA     *A,
      IndexType    ldA,
      const TX     *x,
      IndexType    incX,
      const Beta   &beta,
      TY           *y,
      IndexType    incY)
{
    hbumv(n, k, alpha, false, A, ldA, x, incX, beta, y, incY);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_HBUMV_TCC
