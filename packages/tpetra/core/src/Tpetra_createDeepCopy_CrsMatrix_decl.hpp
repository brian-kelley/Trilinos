#ifndef TPETRA_CREATEDEEPCOPY_CRSMATRIX_DECL_HPP
#define TPETRA_CREATEDEEPCOPY_CRSMATRIX_DECL_HPP

#include "Tpetra_createDeepCopy.hpp"
#include "Tpetra_CrsMatrix_fwd.hpp"
#include "Tpetra_RowMatrix_fwd.hpp"

namespace Tpetra {

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class SC, class LO, class GO, class NT>
CrsMatrix<SC, LO, GO, NT>
createDeepCopy (const RowMatrix<SC, LO, GO, NT>& in);
#else
template<class SC, class NT>
CrsMatrix<SC, NT>
createDeepCopy (const RowMatrix<SC, NT>& in);
#endif

} // namespace Tpetra

#endif // TPETRA_CREATEDEEPCOPY_CRSMATRIX_DECL_HPP
