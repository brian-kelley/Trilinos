// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER

#ifndef MUELU_SHIFTEDLAPLACIANOPERATOR_DEF_HPP
#define MUELU_SHIFTEDLAPLACIANOPERATOR_DEF_HPP

#include "MueLu_ConfigDefs.hpp"

#ifdef HAVE_MUELU_TPETRA

#include <Xpetra_Matrix.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_BlockedCrsMatrix.hpp>
#include <Xpetra_TpetraMultiVector.hpp>
#include <Xpetra_MultiVectorFactory.hpp>

#include "MueLu_ShiftedLaplacianOperator_decl.hpp"
#include "MueLu_Hierarchy.hpp"
#include "MueLu_Utilities.hpp"


namespace MueLu {

// ------------- getDomainMap -----------------------

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >
ShiftedLaplacianOperator<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
template <class Scalar, class Node>
Teuchos::RCP<const Tpetra::Map<Node> >
ShiftedLaplacianOperator<Scalar,Node>::
#endif
getDomainMap () const
{
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  typedef Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> XMatrix;
#else
  typedef Xpetra::Matrix<Scalar, Node> XMatrix;
#endif

  RCP<MueLu::Level> L0 = Hierarchy_->GetLevel (0);
  RCP<XMatrix> A = L0->Get<RCP<XMatrix> > ("A");

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  RCP<Xpetra::BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > tpbA =
    Teuchos::rcp_dynamic_cast<Xpetra::BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >(A);
#else
  RCP<Xpetra::BlockedCrsMatrix<Scalar, Node> > tpbA =
    Teuchos::rcp_dynamic_cast<Xpetra::BlockedCrsMatrix<Scalar, Node> >(A);
#endif
  if (tpbA != Teuchos::null) {
    return Xpetra::toTpetraNonZero (tpbA->getDomainMap ());
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > tpA =
    Utilities<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Op2NonConstTpetraCrs (A);
#else
  RCP<Tpetra::CrsMatrix<Scalar, Node> > tpA =
    Utilities<Scalar, Node>::Op2NonConstTpetraCrs (A);
#endif
  return tpA->getDomainMap ();
}

// ------------- getRangeMap -----------------------

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >
ShiftedLaplacianOperator<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
template <class Scalar, class Node>
Teuchos::RCP<const Tpetra::Map<Node> >
ShiftedLaplacianOperator<Scalar,Node>::
#endif
getRangeMap () const
{
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  typedef Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> XMatrix;
#else
  typedef Xpetra::Matrix<Scalar, Node> XMatrix;
#endif

  RCP<MueLu::Level>  L0 = Hierarchy_->GetLevel(0);
  RCP<XMatrix> A = L0->Get< RCP<XMatrix> >("A");

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  RCP<Xpetra::BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > tpbA =
    Teuchos::rcp_dynamic_cast<Xpetra::BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >(A);
#else
  RCP<Xpetra::BlockedCrsMatrix<Scalar, Node> > tpbA =
    Teuchos::rcp_dynamic_cast<Xpetra::BlockedCrsMatrix<Scalar, Node> >(A);
#endif
  if(tpbA != Teuchos::null)
    return Xpetra::toTpetraNonZero(tpbA->getRangeMap());

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  RCP< Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > tpA =
    Utilities<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Op2NonConstTpetraCrs(A);
#else
  RCP< Tpetra::CrsMatrix<Scalar, Node> > tpA =
    Utilities<Scalar, Node>::Op2NonConstTpetraCrs(A);
#endif
  return tpA->getRangeMap();
}

// ------------- apply -----------------------

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void ShiftedLaplacianOperator<Scalar,LocalOrdinal,GlobalOrdinal,Node>::apply(const Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& X,
                                                                               Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Y,
#else
template <class Scalar, class Node>
void ShiftedLaplacianOperator<Scalar,Node>::apply(const Tpetra::MultiVector<Scalar,Node>& X,
                                                                               Tpetra::MultiVector<Scalar,Node>& Y,
#endif
                                                                               Teuchos::ETransp /* mode */, Scalar /* alpha */, Scalar /* beta */) const {

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  typedef Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>        TMV;
  typedef Xpetra::TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>  XTMV;
#else
  typedef Tpetra::MultiVector<Scalar,Node>        TMV;
  typedef Xpetra::TpetraMultiVector<Scalar,Node>  XTMV;
#endif
  // typedef Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>        XMV; // unused

  TMV & temp_x = const_cast<TMV &>(X);
  const XTMV tX(rcpFromRef(temp_x));
  XTMV       tY(rcpFromRef(Y));

  try {
    tY.putScalar(0.0);
    Hierarchy_->Iterate(tX, tY, cycles_, true);
  }

  catch(std::exception& e) {
    //FIXME add message and rethrow
    std::cerr << "Caught an exception in MueLu::ShiftedLaplacianOperator::ApplyInverse():" << std::endl
                                                                                              << e.what() << std::endl;
  }

  // update solution with 2-grid error correction
  /*if(option_==1) {
    for(int j=0; j<cycles_; j++) {
      RCP<XMV> residual       = MueLu::Utilities<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Residual(*A_, tY, tX);
      RCP<XMV> coarseResidual = Xpetra::MultiVectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(R_->getRangeMap(), tX.getNumVectors());
      RCP<XMV> coarseError    = Xpetra::MultiVectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(R_->getRangeMap(), tX.getNumVectors());
      R_ -> apply(*residual, *coarseResidual, Teuchos::NO_TRANS, (Scalar) 1.0, (Scalar) 0.0);
      RCP<TMV> tcoarseR = MueLu::Utilities<Scalar,LocalOrdinal,GlobalOrdinal,Node>::MV2NonConstTpetraMV(coarseResidual);
      RCP<TMV> tcoarseE = MueLu::Utilities<Scalar,LocalOrdinal,GlobalOrdinal,Node>::MV2NonConstTpetraMV(coarseError);
      BelosLP_ -> setProblem(tcoarseE,tcoarseR);
      BelosSM_ -> solve();
      RCP<XMV> fineError      = Xpetra::MultiVectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(P_->getRangeMap(), tX.getNumVectors());
      XTMV tmpcoarseE(rcpFromRef(*tcoarseE));
      P_ -> apply(tmpcoarseE, *fineError, Teuchos::NO_TRANS, (Scalar) 1.0, (Scalar) 0.0);
      tY.update((Scalar) 1.0, *fineError, (Scalar) 1.0);
    }
  }

  try {
    Hierarchy_->Iterate(tX, tY, 1, false);
  }

  catch(std::exception& e) {
    //FIXME add message and rethrow
    std::cerr << "Caught an exception in MueLu::ShiftedLaplacianOperator::ApplyInverse():" << std::endl
    << e.what() << std::endl;
    }*/

}

// ------------- apply -----------------------
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool ShiftedLaplacianOperator<Scalar,LocalOrdinal,GlobalOrdinal,Node>::hasTransposeApply() const {
#else
template <class Scalar, class Node>
bool ShiftedLaplacianOperator<Scalar,Node>::hasTransposeApply() const {
#endif
  return false;
}

} // namespace
#endif //ifdef HAVE_MUELU_TPETRA

#endif //ifdef MUELU_SHIFTEDLAPLACIANOPERATOR_DEF_HPP
