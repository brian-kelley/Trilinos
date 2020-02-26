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


#ifndef THYRA_TPETRA_VECTOR_SPACE_HPP
#define THYRA_TPETRA_VECTOR_SPACE_HPP


#include "Thyra_TpetraVectorSpace_decl.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"
#include "Thyra_TpetraVector.hpp"
#include "Thyra_TpetraMultiVector.hpp"
#include "Thyra_TpetraEuclideanScalarProd.hpp"
#include "Tpetra_Details_StaticView.hpp"

namespace Thyra {


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::create()
#else
template <class Scalar, class Node>
RCP<TpetraVectorSpace<Scalar,Node> >
TpetraVectorSpace<Scalar,Node>::create()
#endif
{
  const RCP<this_t> vs(new this_t);
  vs->weakSelfPtr_ = vs.create_weak();
  return vs;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::initialize(
  const RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > &tpetraMap
#else
template <class Scalar, class Node>
void TpetraVectorSpace<Scalar,Node>::initialize(
  const RCP<const Tpetra::Map<Node> > &tpetraMap
#endif
  )
{
  comm_ = convertTpetraToThyraComm(tpetraMap->getComm());
  tpetraMap_ = tpetraMap;
  this->updateState(tpetraMap->getGlobalNumElements(),
    !tpetraMap->isDistributed());
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  this->setScalarProd(tpetraEuclideanScalarProd<Scalar,LocalOrdinal,GlobalOrdinal,Node>());
#else
  this->setScalarProd(tpetraEuclideanScalarProd<Scalar,Node>());
#endif
}


// Overridden from VectorSpace


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<VectorBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::createMember() const
#else
TpetraVectorSpace<Scalar,Node>::createMember() const
#endif
{
  return tpetraVector<Scalar>(
    weakSelfPtr_.create_strong().getConst(),
    Teuchos::rcp(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      new Tpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node>(tpetraMap_, false)
#else
      new Tpetra::Vector<Scalar,Node>(tpetraMap_, false)
#endif
      )
    );
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP< MultiVectorBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::createMembers(int numMembers) const
#else
TpetraVectorSpace<Scalar,Node>::createMembers(int numMembers) const
#endif
{
  return tpetraMultiVector<Scalar>(
    weakSelfPtr_.create_strong().getConst(),
    tpetraVectorSpace<Scalar>(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      Tpetra::createLocalMapWithNode<LocalOrdinal, GlobalOrdinal, Node>(
#else
      Tpetra::createLocalMapWithNode<Node>(
#endif
        numMembers, tpetraMap_->getComm()
        )
      ),
    Teuchos::rcp(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      new Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>(
#else
      new Tpetra::MultiVector<Scalar,Node>(
#endif
        tpetraMap_, numMembers, false)
      )
    );
  // ToDo: Create wrapper function to create locally replicated vector space
  // and use it.
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
class CopyTpetraMultiVectorViewBack {
public:
#ifndef TPETRA_ENABLE_TEMPLATE_ORDINALS
  using LocalOrdinal = typename Tpetra::Map<>::local_ordinal_type;
  using GlobalOrdinal = typename Tpetra::Map<>::global_ordinal_type;
#endif
  CopyTpetraMultiVectorViewBack( RCP<MultiVectorBase<Scalar> > mv, const RTOpPack::SubMultiVectorView<Scalar>  &raw_mv )
    :mv_(mv), raw_mv_(raw_mv)
    {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tmv = Teuchos::rcp_dynamic_cast<TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(mv_,true)->getTpetraMultiVector();
#else
      RCP<Tpetra::MultiVector<Scalar,Node> > tmv = Teuchos::rcp_dynamic_cast<TpetraMultiVector<Scalar,Node> >(mv_,true)->getTpetraMultiVector();
#endif
      bool inUse = Teuchos::get_extra_data<bool>(tmv,"inUse");
      TEUCHOS_TEST_FOR_EXCEPTION(inUse,
                                 std::runtime_error,
                                 "Cannot use the cached vector simultaneously more than once.");
      inUse = true;
      Teuchos::set_extra_data(inUse,"inUse",Teuchos::outArg(tmv), Teuchos::POST_DESTROY, false);
    }
  ~CopyTpetraMultiVectorViewBack()
    {
      RTOpPack::ConstSubMultiVectorView<Scalar> smv;
      mv_->acquireDetachedView(Range1D(),Range1D(),&smv);
      RTOpPack::assign_entries<Scalar>( Teuchos::outArg(raw_mv_), smv );
      mv_->releaseDetachedView(&smv);
      bool inUse = false;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tmv = Teuchos::rcp_dynamic_cast<TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(mv_,true)->getTpetraMultiVector();
#else
      RCP<Tpetra::MultiVector<Scalar,Node> > tmv = Teuchos::rcp_dynamic_cast<TpetraMultiVector<Scalar,Node> >(mv_,true)->getTpetraMultiVector();
#endif
      Teuchos::set_extra_data(inUse,"inUse",Teuchos::outArg(tmv), Teuchos::POST_DESTROY, false);
    }
private:
  RCP<MultiVectorBase<Scalar> >               mv_;
  const RTOpPack::SubMultiVectorView<Scalar>  raw_mv_;
};


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
RCP< MultiVectorBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::createCachedMembersView(
#else
TpetraVectorSpace<Scalar,Node>::createCachedMembersView(
#endif
  const RTOpPack::SubMultiVectorView<Scalar> &raw_mv ) const
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_TEST_FOR_EXCEPT( raw_mv.subDim() != this->dim() );
#endif

  // Create a multi-vector
  RCP< MultiVectorBase<Scalar> > mv;
  if (!tpetraMap_->isDistributed()) {

    if (tpetraMV_.is_null() || (tpetraMV_->getNumVectors() != size_t (raw_mv.numSubCols()))) {
      if (!tpetraMV_.is_null())
        // The MV is already allocated. If we are still using it, then very bad things can happen.
      TEUCHOS_TEST_FOR_EXCEPTION(Teuchos::get_extra_data<bool>(tpetraMV_,"inUse"),
                                 std::runtime_error,
                                 "Cannot use the cached vector simultaneously more than once.");
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      using IST = typename Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::impl_scalar_type;
      using DT = typename Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::device_type;
#else
      using IST = typename Tpetra::MultiVector<Scalar,Node>::impl_scalar_type;
      using DT = typename Tpetra::MultiVector<Scalar,Node>::device_type;
#endif
      auto dv = ::Tpetra::Details::getStatic2dDualView<IST, DT> (tpetraMap_->getGlobalNumElements(), raw_mv.numSubCols());
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tpetraMV_ = Teuchos::rcp(new Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>(tpetraMap_, dv));
#else
      tpetraMV_ = Teuchos::rcp(new Tpetra::MultiVector<Scalar,Node>(tpetraMap_, dv));
#endif
      bool inUse = false;
      Teuchos::set_extra_data(inUse,"inUse",Teuchos::outArg(tpetraMV_));
    }

    if (tpetraDomainSpace_.is_null() || raw_mv.numSubCols() != tpetraDomainSpace_->localSubDim())
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tpetraDomainSpace_ = tpetraVectorSpace<Scalar>(Tpetra::createLocalMapWithNode<LocalOrdinal, GlobalOrdinal, Node>(raw_mv.numSubCols(), tpetraMap_->getComm()));
#else
      tpetraDomainSpace_ = tpetraVectorSpace<Scalar>(Tpetra::createLocalMapWithNode<Node>(raw_mv.numSubCols(), tpetraMap_->getComm()));
#endif

    mv = tpetraMultiVector<Scalar>(weakSelfPtr_.create_strong().getConst(), tpetraDomainSpace_, tpetraMV_);
  } else {
    mv = this->createMembers(raw_mv.numSubCols());
    bool inUse = false;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tmv = Teuchos::rcp_dynamic_cast<TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(mv,true)->getTpetraMultiVector();
#else
    RCP<Tpetra::MultiVector<Scalar,Node> > tmv = Teuchos::rcp_dynamic_cast<TpetraMultiVector<Scalar,Node> >(mv,true)->getTpetraMultiVector();
#endif
    Teuchos::set_extra_data(inUse,"inUse",Teuchos::outArg(tmv));
  }
  // Copy initial values in raw_mv into multi-vector
  RTOpPack::SubMultiVectorView<Scalar> smv;
  mv->acquireDetachedView(Range1D(),Range1D(),&smv);
  RTOpPack::assign_entries<Scalar>(
    Ptr<const RTOpPack::SubMultiVectorView<Scalar> >(Teuchos::outArg(smv)),
    raw_mv
    );
  mv->commitDetachedView(&smv);
  // Setup smart pointer to multi-vector to copy view back out just before multi-vector is destroyed
  Teuchos::set_extra_data(
    // We create a duplicate of the RCP, otherwise the ref count does not go to zero.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::rcp(new CopyTpetraMultiVectorViewBack<Scalar,LocalOrdinal,GlobalOrdinal,Node>(Teuchos::rcpFromRef(*mv),raw_mv)),
#else
    Teuchos::rcp(new CopyTpetraMultiVectorViewBack<Scalar,Node>(Teuchos::rcpFromRef(*mv),raw_mv)),
#endif
    "CopyTpetraMultiVectorViewBack",
    Teuchos::outArg(mv),
    Teuchos::PRE_DESTROY
    );
  return mv;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
RCP<const MultiVectorBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::createCachedMembersView(
#else
TpetraVectorSpace<Scalar,Node>::createCachedMembersView(
#endif
  const RTOpPack::ConstSubMultiVectorView<Scalar> &raw_mv ) const
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_TEST_FOR_EXCEPT( raw_mv.subDim() != this->dim() );
#endif
  // Create a multi-vector
  RCP< MultiVectorBase<Scalar> > mv;
  if (!tpetraMap_->isDistributed()) {
    if (tpetraMV_.is_null() || (tpetraMV_->getNumVectors() != size_t (raw_mv.numSubCols()))) {
      if (!tpetraMV_.is_null())
        // The MV is already allocated. If we are still using it, then very bad things can happen.
        TEUCHOS_TEST_FOR_EXCEPTION(Teuchos::get_extra_data<bool>(tpetraMV_,"inUse"),
                                   std::runtime_error,
                                   "Cannot use the cached vector simultaneously more than once.");
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      using IST = typename Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::impl_scalar_type;
      using DT = typename Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::device_type;
#else
      using IST = typename Tpetra::MultiVector<Scalar,Node>::impl_scalar_type;
      using DT = typename Tpetra::MultiVector<Scalar,Node>::device_type;
#endif
      auto dv = ::Tpetra::Details::getStatic2dDualView<IST, DT> (tpetraMap_->getGlobalNumElements(), raw_mv.numSubCols());
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tpetraMV_ = Teuchos::rcp(new Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>(tpetraMap_, dv));
#else
      tpetraMV_ = Teuchos::rcp(new Tpetra::MultiVector<Scalar,Node>(tpetraMap_, dv));
#endif
      bool inUse = false;
      Teuchos::set_extra_data(inUse,"inUse",Teuchos::outArg(tpetraMV_));
    }

    if (tpetraDomainSpace_.is_null() || raw_mv.numSubCols() != tpetraDomainSpace_->localSubDim())
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tpetraDomainSpace_ = tpetraVectorSpace<Scalar>(Tpetra::createLocalMapWithNode<LocalOrdinal, GlobalOrdinal, Node>(raw_mv.numSubCols(), tpetraMap_->getComm()));
#else
      tpetraDomainSpace_ = tpetraVectorSpace<Scalar>(Tpetra::createLocalMapWithNode<Node>(raw_mv.numSubCols(), tpetraMap_->getComm()));
#endif

    mv = tpetraMultiVector<Scalar>(weakSelfPtr_.create_strong().getConst(), tpetraDomainSpace_, tpetraMV_);
  } else {
    mv = this->createMembers(raw_mv.numSubCols());
    bool inUse = false;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tmv = Teuchos::rcp_dynamic_cast<TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(mv,true)->getTpetraMultiVector();
#else
    RCP<Tpetra::MultiVector<Scalar,Node> > tmv = Teuchos::rcp_dynamic_cast<TpetraMultiVector<Scalar,Node> >(mv,true)->getTpetraMultiVector();
#endif
    Teuchos::set_extra_data(inUse,"inUse",Teuchos::outArg(tmv));
  }
  // Copy values in raw_mv into multi-vector
  RTOpPack::SubMultiVectorView<Scalar> smv;
  mv->acquireDetachedView(Range1D(),Range1D(),&smv);
  RTOpPack::assign_entries<Scalar>(
    Ptr<const RTOpPack::SubMultiVectorView<Scalar> >(Teuchos::outArg(smv)),
    raw_mv );
  mv->commitDetachedView(&smv);
  return mv;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::hasInCoreView(
#else
template <class Scalar, class Node>
bool TpetraVectorSpace<Scalar,Node>::hasInCoreView(
#endif
  const Range1D& rng_in, const EViewType viewType, const EStrideType strideType
  ) const
{
  const Range1D rng = full_range(rng_in,0,this->dim()-1);
  const Ordinal l_localOffset = this->localOffset();

  const Ordinal myLocalSubDim = tpetraMap_.is_null () ?
    static_cast<Ordinal> (0) : tpetraMap_->getNodeNumElements ();

  return ( l_localOffset<=rng.lbound() && rng.ubound()<l_localOffset+myLocalSubDim );
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP< const VectorSpaceBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::clone() const
#else
TpetraVectorSpace<Scalar,Node>::clone() const
#endif
{
  return tpetraVectorSpace<Scalar>(tpetraMap_);
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getTpetraMap() const
#else
template <class Scalar, class Node>
RCP<const Tpetra::Map<Node> >
TpetraVectorSpace<Scalar,Node>::getTpetraMap() const
#endif
{
  return tpetraMap_;
}

// Overridden from SpmdVectorSpaceDefaultBase


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<const Teuchos::Comm<Ordinal> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getComm() const
#else
TpetraVectorSpace<Scalar,Node>::getComm() const
#endif
{
  return comm_;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Ordinal TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::localSubDim() const
#else
template <class Scalar, class Node>
Ordinal TpetraVectorSpace<Scalar,Node>::localSubDim() const
#endif
{
  return tpetraMap_.is_null () ? static_cast<Ordinal> (0) :
    static_cast<Ordinal> (tpetraMap_->getNodeNumElements ());
}

// private


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraVectorSpace()
#else
template <class Scalar, class Node>
TpetraVectorSpace<Scalar,Node>::TpetraVectorSpace()
#endif
{
  // The base classes should automatically default initialize to a safe
  // uninitialized state.
}


} // end namespace Thyra


#endif // THYRA_TPETRA_VECTOR_SPACE_HPP
