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
//                    Tobias Wiesner    (tawiesn@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef THYRA_XPETRA_LINEAR_OP_HPP
#define THYRA_XPETRA_LINEAR_OP_HPP

#include "Thyra_XpetraLinearOp_decl.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_TypeNameTraits.hpp"

#include "MueLu_XpetraOperator_decl.hpp"
#include "Xpetra_MapExtractor.hpp"

namespace Thyra {


// Constructors/initializers


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
XpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node>::XpetraLinearOp()
#else
template <class Scalar, class Node>
XpetraLinearOp<Scalar,Node>::XpetraLinearOp()
#endif
{}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void XpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node>::initialize(
#else
template <class Scalar, class Node>
void XpetraLinearOp<Scalar,Node>::initialize(
#endif
  const RCP<const VectorSpaceBase<Scalar> > &rangeSpace,
  const RCP<const VectorSpaceBase<Scalar> > &domainSpace,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &xpetraOperator
#else
  const RCP<Xpetra::Operator<Scalar,Node> > &xpetraOperator
#endif
  )
{
  initializeImpl(rangeSpace, domainSpace, xpetraOperator);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void XpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node>::constInitialize(
#else
template <class Scalar, class Node>
void XpetraLinearOp<Scalar,Node>::constInitialize(
#endif
  const RCP<const VectorSpaceBase<Scalar> > &rangeSpace,
  const RCP<const VectorSpaceBase<Scalar> > &domainSpace,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<const Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &xpetraOperator
#else
  const RCP<const Xpetra::Operator<Scalar,Node> > &xpetraOperator
#endif
  )
{
  initializeImpl(rangeSpace, domainSpace, xpetraOperator);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
XpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getXpetraOperator()
#else
template <class Scalar, class Node>
RCP<Xpetra::Operator<Scalar,Node> >
XpetraLinearOp<Scalar,Node>::getXpetraOperator()
#endif
{
  return xpetraOperator_.getNonconstObj();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
XpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getConstXpetraOperator() const
#else
template <class Scalar, class Node>
RCP<const Xpetra::Operator<Scalar,Node> >
XpetraLinearOp<Scalar,Node>::getConstXpetraOperator() const
#endif
{
  return xpetraOperator_;
}


// Public Overridden functions from LinearOpBase


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<const Thyra::VectorSpaceBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
XpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node>::range() const
#else
XpetraLinearOp<Scalar,Node>::range() const
#endif
{
  return rangeSpace_;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<const Thyra::VectorSpaceBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
XpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node>::domain() const
#else
XpetraLinearOp<Scalar,Node>::domain() const
#endif
{
  return domainSpace_;
}

// Protected Overridden functions from LinearOpBase

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool XpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node>::opSupportedImpl(
#else
template <class Scalar, class Node>
bool XpetraLinearOp<Scalar,Node>::opSupportedImpl(
#endif
  Thyra::EOpTransp M_trans) const
{
  if (is_null(xpetraOperator_))
    return false;

  if (M_trans == NOTRANS)
    return true;

  if (M_trans == CONJ) {
    // For non-complex scalars, CONJ is always supported since it is equivalent to NO_TRANS.
    // For complex scalars, Xpetra does not support conjugation without transposition.
    return !Teuchos::ScalarTraits<Scalar>::isComplex;
  }

  return xpetraOperator_->hasTransposeApply();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void XpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node>::applyImpl(
#else
template <class Scalar, class Node>
void XpetraLinearOp<Scalar,Node>::applyImpl(
#endif
  const Thyra::EOpTransp M_trans,
  const Thyra::MultiVectorBase<Scalar> &X_in,
  const Teuchos::Ptr<Thyra::MultiVectorBase<Scalar> > &Y_inout,
  const Scalar alpha,
  const Scalar beta
  ) const
{
  using Teuchos::rcpFromRef;
  using Teuchos::rcpFromPtr;

  TEUCHOS_TEST_FOR_EXCEPTION(getConstXpetraOperator() == Teuchos::null, MueLu::Exceptions::RuntimeError, "XpetraLinearOp::applyImpl: internal Xpetra::Operator is null.");
  RCP< const Teuchos::Comm< int > > comm = getConstXpetraOperator()->getRangeMap()->getComm();

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tX_in =
      Xpetra::ThyraUtils<Scalar,LocalOrdinal,GlobalOrdinal,Node>::toXpetra(rcpFromRef(X_in), comm);
  RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tY_inout =
      Xpetra::ThyraUtils<Scalar,LocalOrdinal,GlobalOrdinal,Node>::toXpetra(rcpFromPtr(Y_inout), comm);
#else
  const RCP<const Xpetra::MultiVector<Scalar,Node> > tX_in =
      Xpetra::ThyraUtils<Scalar,Node>::toXpetra(rcpFromRef(X_in), comm);
  RCP<Xpetra::MultiVector<Scalar,Node> > tY_inout =
      Xpetra::ThyraUtils<Scalar,Node>::toXpetra(rcpFromPtr(Y_inout), comm);
#endif
  Teuchos::ETransp transp;
  switch (M_trans) {
    case NOTRANS:   transp = Teuchos::NO_TRANS;   break;
    case TRANS:     transp = Teuchos::TRANS;      break;
    case CONJTRANS: transp = Teuchos::CONJ_TRANS; break;
    default: TEUCHOS_TEST_FOR_EXCEPTION(true, MueLu::Exceptions::NotImplemented, "Thyra::XpetraLinearOp::apply. Unknown value for M_trans. Only NOTRANS, TRANS and CONJTRANS are supported.");
  }

  xpetraOperator_->apply(*tX_in, *tY_inout, transp, alpha, beta);

  // check whether Y is a product vector
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  RCP<const Xpetra::MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal,Node> > rgMapExtractor = Teuchos::null;
#else
  RCP<const Xpetra::MapExtractor<Scalar,Node> > rgMapExtractor = Teuchos::null;
#endif
  Teuchos::Ptr<Thyra::ProductMultiVectorBase<Scalar> > prodY_inout =
      Teuchos::ptr_dynamic_cast<Thyra::ProductMultiVectorBase<Scalar> >(Y_inout);
  if(prodY_inout != Teuchos::null) {
    // If Y is a product vector we split up the data from tY and merge them
    // into the product vector. The necessary Xpetra::MapExtractor is extracted
    // from the fine level operator (not this!)

    // get underlying fine level operator (BlockedCrsMatrix)
    // to extract the range MapExtractor
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<MueLu::XpetraOperator<Scalar, LocalOrdinal, GlobalOrdinal,Node> > mueXop =
        Teuchos::rcp_dynamic_cast<MueLu::XpetraOperator<Scalar, LocalOrdinal, GlobalOrdinal,Node> >(xpetraOperator_.getNonconstObj());
#else
    RCP<MueLu::XpetraOperator<Scalar,Node> > mueXop =
        Teuchos::rcp_dynamic_cast<MueLu::XpetraOperator<Scalar,Node> >(xpetraOperator_.getNonconstObj());
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal,Node> > A =
        mueXop->GetHierarchy()->GetLevel(0)->template Get<RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal,Node> > >("A");
#else
    RCP<Xpetra::Matrix<Scalar,Node> > A =
        mueXop->GetHierarchy()->GetLevel(0)->template Get<RCP<Xpetra::Matrix<Scalar,Node> > >("A");
#endif
    TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(A));

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Xpetra::BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal,Node> > bA =
        Teuchos::rcp_dynamic_cast<Xpetra::BlockedCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal,Node> >(A);
#else
    RCP<Xpetra::BlockedCrsMatrix<Scalar,Node> > bA =
        Teuchos::rcp_dynamic_cast<Xpetra::BlockedCrsMatrix<Scalar,Node> >(A);
#endif
    TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(bA));

    rgMapExtractor = bA->getRangeMapExtractor();
    TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(rgMapExtractor));
  }
}


// private


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
template<class XpetraOperator_t>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
void XpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node>::initializeImpl(
#else
void XpetraLinearOp<Scalar,Node>::initializeImpl(
#endif
  const RCP<const VectorSpaceBase<Scalar> > &rangeSpace,
  const RCP<const VectorSpaceBase<Scalar> > &domainSpace,
  const RCP<XpetraOperator_t> &xpetraOperator
  )
{
#ifdef THYRA_DEBUG
  TEUCHOS_ASSERT(nonnull(rangeSpace));
  TEUCHOS_ASSERT(nonnull(domainSpace));
  TEUCHOS_ASSERT(nonnull(xpetraOperator));
#endif
  rangeSpace_ = rangeSpace;
  domainSpace_ = domainSpace;
  xpetraOperator_ = xpetraOperator;
}


} // namespace Thyra


#endif  // THYRA_XPETRA_LINEAR_OP_HPP
