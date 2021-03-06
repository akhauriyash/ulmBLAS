#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOTXAXPYF_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOTXAXPYF_TCC 1

#include <type_traits>
#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/config/fusefactor.h>
#include <ulmblas/level1extensions/kernel/ref/dotxaxpyf.h>

#ifdef DDOTXAXPYF_FUSEFACTOR
#undef DDOTXAXPYF_FUSEFACTOR
#endif

#define DDOTXAXPYF_FUSEFACTOR  2

namespace ulmBLAS { namespace ref {

template <typename IndexType, typename Alpha, typename VA, typename MX,
          typename VY, typename VZ, typename Rho>
void
dotxaxpyf(IndexType      n,
          bool           conjX,
          bool           conjXt,
          bool           conjY,
          const Alpha    &alpha,
          const VA       *a,
          IndexType      incA,
          const MX       *X,
          IndexType      incRowX,
          IndexType      incColX,
          const VY       *y,
          IndexType      incY,
          VZ             *z,
          IndexType      incZ,
          Rho            *rho,
          IndexType      incRho)
{
    typedef decltype(Alpha(0)*VA(0)*MX(0)*VY(0)*VZ(0))  T;

    const IndexType bf = FuseFactor<T>::dotxaxpyf;

    for (IndexType l=0; l<bf; ++l) {
        rho[l*incRho] = 0;
    }

    if (n<=0) {
        return;
    }

    for (IndexType i=0; i<n; ++i) {
        const VY y_i = conjugate(y[i*incY], conjY);

        for (IndexType l=0; l<bf; ++l) {
            const MX x_il  = conjugate(X[i*incRowX+l*incColX], conjX);
            const MX xt_il = conjugate(X[i*incRowX+l*incColX], conjXt);

            rho[l*incRho] += xt_il * y_i;
            z[i*incZ]     += alpha*a[l*incA]*x_il;
        }
    }
}

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOTXAXPYF_TCC 1
