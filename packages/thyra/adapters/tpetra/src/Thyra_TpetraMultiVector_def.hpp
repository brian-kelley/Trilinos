// @HEADER
// ***********************************************************************
//
//    Thyra: Interfaces and Support for Abstract Numerical Algorithms
//                 Copyright (2004) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact Roscoe A. Bartlett (bartlettra@ornl.gov)
//
// ***********************************************************************
// @HEADER

#ifndef THYRA_TPETRA_MULTIVECTOR_HPP
#define THYRA_TPETRA_MULTIVECTOR_HPP

#include "Thyra_TpetraMultiVector_decl.hpp"
#include "Thyra_TpetraVectorSpace.hpp"
#include "Thyra_TpetraVector.hpp"
#include "Teuchos_Assert.hpp"


namespace Thyra {


// Constructors/initializers/accessors


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraMultiVector()
#else
template <class Scalar, class Node>
TpetraMultiVector<Scalar,Node>::TpetraMultiVector()
#endif
{}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::initialize(
  const RCP<const TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraVectorSpace,
#else
template <class Scalar, class Node>
void TpetraMultiVector<Scalar,Node>::initialize(
  const RCP<const TpetraVectorSpace<Scalar,Node> > &tpetraVectorSpace,
#endif
  const RCP<const ScalarProdVectorSpaceBase<Scalar> > &domainSpace,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraMultiVector
#else
  const RCP<Tpetra::MultiVector<Scalar,Node> > &tpetraMultiVector
#endif
  )
{
  initializeImpl(tpetraVectorSpace, domainSpace, tpetraMultiVector);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::constInitialize(
  const RCP<const TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraVectorSpace,
#else
template <class Scalar, class Node>
void TpetraMultiVector<Scalar,Node>::constInitialize(
  const RCP<const TpetraVectorSpace<Scalar,Node> > &tpetraVectorSpace,
#endif
  const RCP<const ScalarProdVectorSpaceBase<Scalar> > &domainSpace,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<const Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraMultiVector
#else
  const RCP<const Tpetra::MultiVector<Scalar,Node> > &tpetraMultiVector
#endif
  )
{
  initializeImpl(tpetraVectorSpace, domainSpace, tpetraMultiVector);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getTpetraMultiVector()
#else
template <class Scalar, class Node>
RCP<Tpetra::MultiVector<Scalar,Node> >
TpetraMultiVector<Scalar,Node>::getTpetraMultiVector()
#endif
{
  return tpetraMultiVector_.getNonconstObj();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getConstTpetraMultiVector() const
#else
template <class Scalar, class Node>
RCP<const Tpetra::MultiVector<Scalar,Node> >
TpetraMultiVector<Scalar,Node>::getConstTpetraMultiVector() const
#endif
{
  return tpetraMultiVector_;
}


// Overridden public functions form MultiVectorAdapterBase


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP< const ScalarProdVectorSpaceBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::domainScalarProdVecSpc() const
#else
TpetraMultiVector<Scalar,Node>::domainScalarProdVecSpc() const
#endif
{
  return domainSpace_;
}


// Overridden protected functions from MultiVectorBase


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::assignImpl(Scalar alpha)
#else
TpetraMultiVector<Scalar,Node>::assignImpl(Scalar alpha)
#endif
{
  tpetraMultiVector_.getNonconstObj()->putScalar(alpha);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
template <class Scalar, class Node>
void TpetraMultiVector<Scalar,Node>::
#endif
assignMultiVecImpl(const MultiVectorBase<Scalar>& mv)
{
  auto tmv = this->getConstTpetraMultiVector(Teuchos::rcpFromRef(mv));

  // If the cast succeeded, call Tpetra directly.
  // Otherwise, fall back to the RTOp implementation.
  if (nonnull(tmv)) {
    tpetraMultiVector_.getNonconstObj()->assign(*tmv);
  } else {
    // This version will require/modify the host view of this vector.
    tpetraMultiVector_.getNonconstObj()->sync_host ();
    tpetraMultiVector_.getNonconstObj()->modify_host ();
    MultiVectorDefaultBase<Scalar>::assignMultiVecImpl(mv);
  }
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::scaleImpl(Scalar alpha)
#else
TpetraMultiVector<Scalar,Node>::scaleImpl(Scalar alpha)
#endif
{
  tpetraMultiVector_.getNonconstObj()->scale(alpha);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::updateImpl(
#else
template <class Scalar, class Node>
void TpetraMultiVector<Scalar,Node>::updateImpl(
#endif
  Scalar alpha,
  const MultiVectorBase<Scalar>& mv
  )
{
  auto tmv = this->getConstTpetraMultiVector(Teuchos::rcpFromRef(mv));

  // If the cast succeeded, call Tpetra directly.
  // Otherwise, fall back to the RTOp implementation.
  if (nonnull(tmv)) {
    typedef Teuchos::ScalarTraits<Scalar> ST;
    tpetraMultiVector_.getNonconstObj()->update(alpha, *tmv, ST::one());
  } else {
    // This version will require/modify the host view of this vector.
    tpetraMultiVector_.getNonconstObj()->sync_host ();
    tpetraMultiVector_.getNonconstObj()->modify_host ();
    MultiVectorDefaultBase<Scalar>::updateImpl(alpha, mv);
  }
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::linearCombinationImpl(
#else
template <class Scalar, class Node>
void TpetraMultiVector<Scalar,Node>::linearCombinationImpl(
#endif
  const ArrayView<const Scalar>& alpha,
  const ArrayView<const Ptr<const MultiVectorBase<Scalar> > >& mv,
  const Scalar& beta
  )
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT_EQUALITY(alpha.size(), mv.size());
#endif

  // Try to cast mv to an array of this type
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  typedef Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> TMV;
#else
  typedef Tpetra::MultiVector<Scalar,Node> TMV;
#endif
  Teuchos::Array<RCP<const TMV> > tmvs(mv.size());
  RCP<const TMV> tmv;
  bool allCastsSuccessful = true;
  {
    auto mvIter = mv.begin();
    auto tmvIter = tmvs.begin();
    for (; mvIter != mv.end(); ++mvIter, ++tmvIter) {
      tmv = this->getConstTpetraMultiVector(Teuchos::rcpFromPtr(*mvIter));
      if (nonnull(tmv)) {
        *tmvIter = tmv;
      } else {
        allCastsSuccessful = false;
        break;
      }
    }
  }

  // If casts succeeded, or input arrays are size 0, call Tpetra directly.
  // Otherwise, fall back to the RTOp implementation.
  auto len = tmvs.size();
  if (len == 0) {
    tpetraMultiVector_.getNonconstObj()->scale(beta);
  } else if (len == 1 && allCastsSuccessful) {
    tpetraMultiVector_.getNonconstObj()->update(alpha[0], *tmvs[0], beta);
  } else if (len == 2 && allCastsSuccessful) {
    tpetraMultiVector_.getNonconstObj()->update(alpha[0], *tmvs[0], alpha[1], *tmvs[1], beta);
  } else if (allCastsSuccessful) {
    typedef Teuchos::ScalarTraits<Scalar> ST;
    auto tmvIter = tmvs.begin();
    auto alphaIter = alpha.begin();

    // Check if any entry of tmvs aliases this object's wrapped vector.
    // If so, replace that entry in the array with a copy.
    tmv = Teuchos::null;
    for (; tmvIter != tmvs.end(); ++tmvIter) {
      if (tmvIter->getRawPtr() == tpetraMultiVector_.getConstObj().getRawPtr()) {
        if (tmv.is_null()) {
          tmv = Teuchos::rcp(new TMV(*tpetraMultiVector_.getConstObj(), Teuchos::Copy));
        }
        *tmvIter = tmv;
      }
    }
    tmvIter = tmvs.begin();

    // We add two MVs at a time, so only scale if even num MVs,
    // and additionally do the first addition if odd num MVs.
    if ((tmvs.size() % 2) == 0) {
      tpetraMultiVector_.getNonconstObj()->scale(beta);
    } else {
      tpetraMultiVector_.getNonconstObj()->update(*alphaIter, *(*tmvIter), beta);
      ++tmvIter;
      ++alphaIter;
    }
    for (; tmvIter != tmvs.end(); tmvIter+=2, alphaIter+=2) {
      tpetraMultiVector_.getNonconstObj()->update(
        *alphaIter, *(*tmvIter), *(alphaIter+1), *(*(tmvIter+1)), ST::one());
    }
  } else {
    // This version will require/modify the host view of this vector.
    tpetraMultiVector_.getNonconstObj()->sync_host ();
    tpetraMultiVector_.getNonconstObj()->modify_host ();
    MultiVectorDefaultBase<Scalar>::linearCombinationImpl(alpha, mv, beta);
  }
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::dotsImpl(
#else
template <class Scalar, class Node>
void TpetraMultiVector<Scalar,Node>::dotsImpl(
#endif
    const MultiVectorBase<Scalar>& mv,
    const ArrayView<Scalar>& prods
    ) const
{
  auto tmv = this->getConstTpetraMultiVector(Teuchos::rcpFromRef(mv));

  // If the cast succeeded, call Tpetra directly.
  // Otherwise, fall back to the RTOp implementation.
  if (nonnull(tmv)) {
    tpetraMultiVector_.getConstObj()->dot(*tmv, prods);
  } else {
    // This version will require/modify the host view of this vector.
    tpetraMultiVector_.getNonconstObj()->sync_host ();
    tpetraMultiVector_.getNonconstObj()->modify_host ();
    MultiVectorDefaultBase<Scalar>::dotsImpl(mv, prods);
  }
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::norms1Impl(
#else
template <class Scalar, class Node>
void TpetraMultiVector<Scalar,Node>::norms1Impl(
#endif
  const ArrayView<typename ScalarTraits<Scalar>::magnitudeType>& norms
  ) const
{
  tpetraMultiVector_.getConstObj()->norm1(norms);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::norms2Impl(
#else
template <class Scalar, class Node>
void TpetraMultiVector<Scalar,Node>::norms2Impl(
#endif
    const ArrayView<typename ScalarTraits<Scalar>::magnitudeType>& norms
    ) const
{
  tpetraMultiVector_.getConstObj()->norm2(norms);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::normsInfImpl(
#else
template <class Scalar, class Node>
void TpetraMultiVector<Scalar,Node>::normsInfImpl(
#endif
  const ArrayView<typename ScalarTraits<Scalar>::magnitudeType>& norms
  ) const
{
  tpetraMultiVector_.getConstObj()->normInf(norms);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<const VectorBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::colImpl(Ordinal j) const
#else
TpetraMultiVector<Scalar,Node>::colImpl(Ordinal j) const
#endif
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT_IN_RANGE_UPPER_EXCLUSIVE(j, 0, this->domain()->dim());
#endif
  return constTpetraVector<Scalar>(
    tpetraVectorSpace_,
    tpetraMultiVector_->getVector(j)
    );
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<VectorBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::nonconstColImpl(Ordinal j)
#else
TpetraMultiVector<Scalar,Node>::nonconstColImpl(Ordinal j)
#endif
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT_IN_RANGE_UPPER_EXCLUSIVE(j, 0, this->domain()->dim());
#endif
  return tpetraVector<Scalar>(
    tpetraVectorSpace_,
    tpetraMultiVector_.getNonconstObj()->getVectorNonConst(j)
    );
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<const MultiVectorBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::contigSubViewImpl(
#else
TpetraMultiVector<Scalar,Node>::contigSubViewImpl(
#endif
  const Range1D& col_rng_in
  ) const
{
#ifdef THYRA_DEFAULT_SPMD_MULTI_VECTOR_VERBOSE_TO_ERROR_OUT
  std::cerr << "\nTpetraMultiVector::subView(Range1D) const called!\n";
#endif
  const Range1D colRng = this->validateColRange(col_rng_in);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<const Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tpetraView =
#else
  const RCP<const Tpetra::MultiVector<Scalar,Node> > tpetraView =
#endif
    this->getConstTpetraMultiVector()->subView(colRng);

  const RCP<const ScalarProdVectorSpaceBase<Scalar> > viewDomainSpace =
    tpetraVectorSpace<Scalar>(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        Tpetra::createLocalMapWithNode<LocalOrdinal,GlobalOrdinal,Node>(
#else
        Tpetra::createLocalMapWithNode<Node>(
#endif
          tpetraView->getNumVectors(),
          tpetraView->getMap()->getComm()
          )
        );

  return constTpetraMultiVector(
      tpetraVectorSpace_,
      viewDomainSpace,
      tpetraView
      );
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<MultiVectorBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::nonconstContigSubViewImpl(
#else
TpetraMultiVector<Scalar,Node>::nonconstContigSubViewImpl(
#endif
  const Range1D& col_rng_in
  )
{
#ifdef THYRA_DEFAULT_SPMD_MULTI_VECTOR_VERBOSE_TO_ERROR_OUT
  std::cerr << "\nTpetraMultiVector::subView(Range1D) called!\n";
#endif
  const Range1D colRng = this->validateColRange(col_rng_in);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tpetraView =
#else
  const RCP<Tpetra::MultiVector<Scalar,Node> > tpetraView =
#endif
    this->getTpetraMultiVector()->subViewNonConst(colRng);

  const RCP<const ScalarProdVectorSpaceBase<Scalar> > viewDomainSpace =
    tpetraVectorSpace<Scalar>(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        Tpetra::createLocalMapWithNode<LocalOrdinal,GlobalOrdinal,Node>(
#else
        Tpetra::createLocalMapWithNode<Node>(
#endif
          tpetraView->getNumVectors(),
          tpetraView->getMap()->getComm()
          )
        );

  return tpetraMultiVector(
      tpetraVectorSpace_,
      viewDomainSpace,
      tpetraView
      );
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<const MultiVectorBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::nonContigSubViewImpl(
#else
TpetraMultiVector<Scalar,Node>::nonContigSubViewImpl(
#endif
  const ArrayView<const int>& cols_in
  ) const
{
#ifdef THYRA_DEFAULT_SPMD_MULTI_VECTOR_VERBOSE_TO_ERROR_OUT
  std::cerr << "\nTpetraMultiVector::subView(ArrayView) const called!\n";
#endif
  // Tpetra wants col indices as size_t
  Array<std::size_t> cols(cols_in.size());
  for (Array<std::size_t>::size_type i = 0; i < cols.size(); ++i)
    cols[i] = static_cast<std::size_t>(cols_in[i]);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<const Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tpetraView =
#else
  const RCP<const Tpetra::MultiVector<Scalar,Node> > tpetraView =
#endif
    this->getConstTpetraMultiVector()->subView(cols());

  const RCP<const ScalarProdVectorSpaceBase<Scalar> > viewDomainSpace =
    tpetraVectorSpace<Scalar>(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        Tpetra::createLocalMapWithNode<LocalOrdinal,GlobalOrdinal,Node>(
#else
        Tpetra::createLocalMapWithNode<Node>(
#endif
          tpetraView->getNumVectors(),
          tpetraView->getMap()->getComm()
          )
        );

  return constTpetraMultiVector(
      tpetraVectorSpace_,
      viewDomainSpace,
      tpetraView
      );
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<MultiVectorBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::nonconstNonContigSubViewImpl(
#else
TpetraMultiVector<Scalar,Node>::nonconstNonContigSubViewImpl(
#endif
  const ArrayView<const int>& cols_in
  )
{
#ifdef THYRA_DEFAULT_SPMD_MULTI_VECTOR_VERBOSE_TO_ERROR_OUT
  std::cerr << "\nTpetraMultiVector::subView(ArrayView) called!\n";
#endif
  // Tpetra wants col indices as size_t
  Array<std::size_t> cols(cols_in.size());
  for (Array<std::size_t>::size_type i = 0; i < cols.size(); ++i)
    cols[i] = static_cast<std::size_t>(cols_in[i]);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tpetraView =
#else
  const RCP<Tpetra::MultiVector<Scalar,Node> > tpetraView =
#endif
    this->getTpetraMultiVector()->subViewNonConst(cols());

  const RCP<const ScalarProdVectorSpaceBase<Scalar> > viewDomainSpace =
    tpetraVectorSpace<Scalar>(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        Tpetra::createLocalMapWithNode<LocalOrdinal,GlobalOrdinal,Node>(
#else
        Tpetra::createLocalMapWithNode<Node>(
#endif
          tpetraView->getNumVectors(),
          tpetraView->getMap()->getComm()
          )
        );

  return tpetraMultiVector(
      tpetraVectorSpace_,
      viewDomainSpace,
      tpetraView
      );
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
template <class Scalar, class Node>
void TpetraMultiVector<Scalar,Node>::
#endif
mvMultiReductApplyOpImpl(
  const RTOpPack::RTOpT<Scalar> &primary_op,
  const ArrayView<const Ptr<const MultiVectorBase<Scalar> > > &multi_vecs,
  const ArrayView<const Ptr<MultiVectorBase<Scalar> > > &targ_multi_vecs,
  const ArrayView<const Ptr<RTOpPack::ReductTarget> > &reduct_objs,
  const Ordinal primary_global_offset
  ) const
{
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  typedef TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> TMV;
#else
  typedef TpetraMultiVector<Scalar,Node> TMV;
#endif

  // Sync any non-target Tpetra MVs to host space
  for (auto itr = multi_vecs.begin(); itr != multi_vecs.end(); ++itr) {
    Ptr<const TMV> tmv = Teuchos::ptr_dynamic_cast<const TMV>(*itr);
    if (nonnull(tmv)) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      Teuchos::rcp_const_cast<Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(
#else
      Teuchos::rcp_const_cast<Tpetra::MultiVector<Scalar,Node> >(
#endif
      tmv->getConstTpetraMultiVector())-> sync_host ();
    }
  }

  // Sync any target Tpetra MVs and mark modified
  for (auto itr = targ_multi_vecs.begin(); itr != targ_multi_vecs.end(); ++itr) {
    Ptr<TMV> tmv = Teuchos::ptr_dynamic_cast<TMV>(*itr);
    if (nonnull(tmv)) {
      tmv->getTpetraMultiVector()->sync_host ();
      tmv->getTpetraMultiVector()->modify_host ();
    }
  }

  MultiVectorAdapterBase<Scalar>::mvMultiReductApplyOpImpl(
    primary_op, multi_vecs, targ_multi_vecs, reduct_objs, primary_global_offset);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
template <class Scalar, class Node>
void TpetraMultiVector<Scalar,Node>::
#endif
acquireDetachedMultiVectorViewImpl(
  const Range1D &rowRng,
  const Range1D &colRng,
  RTOpPack::ConstSubMultiVectorView<Scalar>* sub_mv
  ) const
{
  // Only viewing data, so just sync dual view to host space
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  typedef typename Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> TMV;
#else
  typedef typename Tpetra::MultiVector<Scalar,Node> TMV;
#endif
  Teuchos::rcp_const_cast<TMV>(
    tpetraMultiVector_.getConstObj())->sync_host ();

  SpmdMultiVectorDefaultBase<Scalar>::
    acquireDetachedMultiVectorViewImpl(rowRng, colRng, sub_mv);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
template <class Scalar, class Node>
void TpetraMultiVector<Scalar,Node>::
#endif
acquireNonconstDetachedMultiVectorViewImpl(
  const Range1D &rowRng,
  const Range1D &colRng,
  RTOpPack::SubMultiVectorView<Scalar>* sub_mv
  )
{
  // Sync to host and mark as modified
  tpetraMultiVector_.getNonconstObj()->sync_host ();
  tpetraMultiVector_.getNonconstObj()->modify_host ();

  SpmdMultiVectorDefaultBase<Scalar>::
    acquireNonconstDetachedMultiVectorViewImpl(rowRng, colRng, sub_mv);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
template <class Scalar, class Node>
void TpetraMultiVector<Scalar,Node>::
#endif
commitNonconstDetachedMultiVectorViewImpl(
  RTOpPack::SubMultiVectorView<Scalar>* sub_mv
  )
{
  SpmdMultiVectorDefaultBase<Scalar>::
    commitNonconstDetachedMultiVectorViewImpl(sub_mv);

  // Sync changes from host view to execution space
  typedef typename Tpetra::MultiVector<
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Scalar,LocalOrdinal,GlobalOrdinal,Node>::execution_space execution_space;
#else
    Scalar,Node>::execution_space execution_space;
#endif
  tpetraMultiVector_.getNonconstObj()->template sync<execution_space>();
}


/* ToDo: Implement these?


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const MultiVectorBase<Scalar> >
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::nonContigSubViewImpl(
  const ArrayView<const int> &cols
  ) const
{
  THYRA_DEBUG_ASSERT_MV_COLS("nonContigSubViewImpl(cols)", cols);
  const int numCols = cols.size();
  const ArrayRCP<Scalar> localValuesView = createContiguousCopy(cols);
  return defaultSpmdMultiVector<Scalar>(
    spmdRangeSpace_,
    createSmallScalarProdVectorSpaceBase<Scalar>(spmdRangeSpace_, numCols),
    localValuesView
    );
}


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<MultiVectorBase<Scalar> >
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::nonconstNonContigSubViewImpl(
  const ArrayView<const int> &cols )
{
  THYRA_DEBUG_ASSERT_MV_COLS("nonContigSubViewImpl(cols)", cols);
  const int numCols = cols.size();
  const ArrayRCP<Scalar> localValuesView = createContiguousCopy(cols);
  const Ordinal localSubDim = spmdRangeSpace_->localSubDim();
  RCP<CopyBackSpmdMultiVectorEntries<Scalar> > copyBackView =
    copyBackSpmdMultiVectorEntries<Scalar>(cols, localValuesView.getConst(),
      localSubDim, localValues_.create_weak(), leadingDim_);
  return Teuchos::rcpWithEmbeddedObjPreDestroy(
    new TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>(
      spmdRangeSpace_,
      createSmallScalarProdVectorSpaceBase<Scalar>(spmdRangeSpace_, numCols),
      localValuesView),
    copyBackView
    );
}

*/


// Overridden protected members from SpmdMultiVectorBase


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<const SpmdVectorSpaceBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::spmdSpaceImpl() const
#else
TpetraMultiVector<Scalar,Node>::spmdSpaceImpl() const
#endif
{
  return tpetraVectorSpace_;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getNonconstLocalMultiVectorDataImpl(
#else
template <class Scalar, class Node>
void TpetraMultiVector<Scalar,Node>::getNonconstLocalMultiVectorDataImpl(
#endif
  const Ptr<ArrayRCP<Scalar> > &localValues, const Ptr<Ordinal> &leadingDim
  )
{
  *localValues = tpetraMultiVector_.getNonconstObj()->get1dViewNonConst();
  *leadingDim = tpetraMultiVector_->getStride();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getLocalMultiVectorDataImpl(
#else
template <class Scalar, class Node>
void TpetraMultiVector<Scalar,Node>::getLocalMultiVectorDataImpl(
#endif
  const Ptr<ArrayRCP<const Scalar> > &localValues, const Ptr<Ordinal> &leadingDim
  ) const
{
  *localValues = tpetraMultiVector_->get1dView();
  *leadingDim = tpetraMultiVector_->getStride();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::euclideanApply(
#else
template <class Scalar, class Node>
void TpetraMultiVector<Scalar,Node>::euclideanApply(
#endif
  const EOpTransp M_trans,
  const MultiVectorBase<Scalar> &X,
  const Ptr<MultiVectorBase<Scalar> > &Y,
  const Scalar alpha,
  const Scalar beta
  ) const
{
  // Try to extract Tpetra objects from X and Y
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  typedef Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> TMV;
#else
  typedef Tpetra::MultiVector<Scalar,Node> TMV;
#endif
  Teuchos::RCP<const TMV> X_tpetra = this->getConstTpetraMultiVector(Teuchos::rcpFromRef(X));
  Teuchos::RCP<TMV> Y_tpetra = this->getTpetraMultiVector(Teuchos::rcpFromPtr(Y));

  // If the cast succeeded, call Tpetra directly.
  // Otherwise, fall back to the default implementation.
  if (nonnull(X_tpetra) && nonnull(Y_tpetra)) {
    // Sync everything to the execution space
    typedef typename TMV::execution_space execution_space;
    Teuchos::rcp_const_cast<TMV>(X_tpetra)->template sync<execution_space>();
    Y_tpetra->template sync<execution_space>();
    Teuchos::rcp_const_cast<TMV>(
      tpetraMultiVector_.getConstObj())->template sync<execution_space>();

    typedef Teuchos::ScalarTraits<Scalar> ST;
    TEUCHOS_TEST_FOR_EXCEPTION(ST::isComplex && (M_trans == CONJ),
      std::logic_error,
      "Error, conjugation without transposition is not allowed for complex scalar types!");

    Teuchos::ETransp trans = Teuchos::NO_TRANS;
    switch (M_trans) {
      case NOTRANS:
        trans = Teuchos::NO_TRANS;
        break;
      case CONJ:
        trans = Teuchos::NO_TRANS;
        break;
      case TRANS:
        trans = Teuchos::TRANS;
        break;
      case CONJTRANS:
        trans = Teuchos::CONJ_TRANS;
        break;
    }

    Y_tpetra->template modify<execution_space>();
    Y_tpetra->multiply(trans, Teuchos::NO_TRANS, alpha, *tpetraMultiVector_.getConstObj(), *X_tpetra, beta);
  } else {
    Teuchos::rcp_const_cast<TMV>(
      tpetraMultiVector_.getConstObj())->sync_host ();
    SpmdMultiVectorDefaultBase<Scalar>::euclideanApply(M_trans, X, Y, alpha, beta);
  }

}

// private


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
template<class TpetraMultiVector_t>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::initializeImpl(
  const RCP<const TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraVectorSpace,
#else
void TpetraMultiVector<Scalar,Node>::initializeImpl(
  const RCP<const TpetraVectorSpace<Scalar,Node> > &tpetraVectorSpace,
#endif
  const RCP<const ScalarProdVectorSpaceBase<Scalar> > &domainSpace,
  const RCP<TpetraMultiVector_t> &tpetraMultiVector
  )
{
#ifdef THYRA_DEBUG
  TEUCHOS_ASSERT(nonnull(tpetraVectorSpace));
  TEUCHOS_ASSERT(nonnull(domainSpace));
  TEUCHOS_ASSERT(nonnull(tpetraMultiVector));
  // ToDo: Check to make sure that tpetraMultiVector is compatible with
  // tpetraVectorSpace.
#endif
  tpetraVectorSpace_ = tpetraVectorSpace;
  domainSpace_ = domainSpace;
  tpetraMultiVector_.initialize(tpetraMultiVector);
  this->updateSpmdSpace();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
template <class Scalar, class Node>
RCP<Tpetra::MultiVector<Scalar,Node> >
TpetraMultiVector<Scalar,Node>::
#endif
getTpetraMultiVector(const RCP<MultiVectorBase<Scalar> >& mv) const
{
  using Teuchos::rcp_dynamic_cast;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  typedef Thyra::TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> TMV;
  typedef Thyra::TpetraVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> TV;
#else
  typedef Thyra::TpetraMultiVector<Scalar,Node> TMV;
  typedef Thyra::TpetraVector<Scalar,Node> TV;
#endif

  RCP<TMV> tmv = rcp_dynamic_cast<TMV>(mv);
  if (nonnull(tmv)) {
    return tmv->getTpetraMultiVector();
  }

  RCP<TV> tv = rcp_dynamic_cast<TV>(mv);
  if (nonnull(tv)) {
    return tv->getTpetraVector();
  }

  return Teuchos::null;
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
template <class Scalar, class Node>
RCP<const Tpetra::MultiVector<Scalar,Node> >
TpetraMultiVector<Scalar,Node>::
#endif
getConstTpetraMultiVector(const RCP<const MultiVectorBase<Scalar> >& mv) const
{
  using Teuchos::rcp_dynamic_cast;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  typedef Thyra::TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> TMV;
  typedef Thyra::TpetraVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> TV;
#else
  typedef Thyra::TpetraMultiVector<Scalar,Node> TMV;
  typedef Thyra::TpetraVector<Scalar,Node> TV;
#endif

  RCP<const TMV> tmv = rcp_dynamic_cast<const TMV>(mv);
  if (nonnull(tmv)) {
    return tmv->getConstTpetraMultiVector();
  }

  RCP<const TV> tv = rcp_dynamic_cast<const TV>(mv);
  if (nonnull(tv)) {
    return tv->getConstTpetraVector();
  }

  return Teuchos::null;
}


} // end namespace Thyra


#endif // THYRA_TPETRA_MULTIVECTOR_HPP
