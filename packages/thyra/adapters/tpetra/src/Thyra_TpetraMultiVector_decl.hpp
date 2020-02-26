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

#ifndef THYRA_TPETRA_MULTIVECTOR_DECL_HPP
#define THYRA_TPETRA_MULTIVECTOR_DECL_HPP

#include "Thyra_SpmdMultiVectorDefaultBase.hpp"
#include "Thyra_TpetraVectorSpace_decl.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Teuchos_ConstNonconstObjectContainer.hpp"


namespace Thyra {


/** \brief Concrete implementation of Thyra::MultiVector in terms of
 * Tpetra::MultiVector.
 *
 * \todo Finish documentation!
 *
 * \ingroup Tpetra_Thyra_Op_Vec_adapters_grp
 */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
class TpetraMultiVector
  : virtual public SpmdMultiVectorDefaultBase<Scalar>
{
public:

#ifndef TPETRA_ENABLE_TEMPLATE_ORDINALS
  using LocalOrdinal = typename Tpetra::Map<>::local_ordinal_type;
  using GlobalOrdinal = typename Tpetra::Map<>::global_ordinal_type;
#endif
  /** @name Constructors/initializers/accessors */
  //@{

  /// Construct to uninitialized
  TpetraMultiVector();

  /** \brief Initialize.
   */
  void initialize(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    const RCP<const TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraVectorSpace,
#else
    const RCP<const TpetraVectorSpace<Scalar,Node> > &tpetraVectorSpace,
#endif
    const RCP<const ScalarProdVectorSpaceBase<Scalar> > &domainSpace,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    const RCP<Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraMultiVector
#else
    const RCP<Tpetra::MultiVector<Scalar,Node> > &tpetraMultiVector
#endif
    );

  /** \brief Initialize.
   */
  void constInitialize(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    const RCP<const TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraVectorSpace,
#else
    const RCP<const TpetraVectorSpace<Scalar,Node> > &tpetraVectorSpace,
#endif
    const RCP<const ScalarProdVectorSpaceBase<Scalar> > &domainSpace,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    const RCP<const Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraMultiVector
#else
    const RCP<const Tpetra::MultiVector<Scalar,Node> > &tpetraMultiVector
#endif
    );

  /** \brief Extract the underlying non-const Tpetra::MultiVector object.*/
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  RCP<Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
#else
  RCP<Tpetra::MultiVector<Scalar,Node> >
#endif
  getTpetraMultiVector();

  /** \brief Extract the underlying const Tpetra::MultiVector object.*/
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  RCP<const Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
#else
  RCP<const Tpetra::MultiVector<Scalar,Node> >
#endif
  getConstTpetraMultiVector() const;

  //@}

  /** @name Overridden public functions form MultiVectorAdapterBase */
  //@{
  /** \brief . */
  RCP< const ScalarProdVectorSpaceBase<Scalar> >
  domainScalarProdVecSpc() const;
  //@}

protected:

  /** @name Overridden protected functions from MultiVectorBase */
  //@{
  /** \brief . */
  virtual void assignImpl(Scalar alpha);

  /** \brief . */
  virtual void assignMultiVecImpl(const MultiVectorBase<Scalar>& mv);

  /** \brief . */
  virtual void scaleImpl(Scalar alpha);

  /** \brief . */
  virtual void updateImpl(
    Scalar alpha,
    const MultiVectorBase<Scalar>& mv
    );

  /** \brief . */
  virtual void linearCombinationImpl(
    const ArrayView<const Scalar>& alpha,
    const ArrayView<const Ptr<const MultiVectorBase<Scalar> > >& mv,
    const Scalar& beta
    );

  /** \brief . */
  virtual void dotsImpl(
    const MultiVectorBase<Scalar>& mv,
    const ArrayView<Scalar>& prods
    ) const;

  /** \brief . */
  virtual void norms1Impl(
    const ArrayView<typename ScalarTraits<Scalar>::magnitudeType>& norms
    ) const;

  /** \brief . */
  virtual void norms2Impl(
    const ArrayView<typename ScalarTraits<Scalar>::magnitudeType>& norms
    ) const;

  /** \brief . */
  virtual void normsInfImpl(
    const ArrayView<typename ScalarTraits<Scalar>::magnitudeType>& norms
    ) const;

  /** \brief . */
  RCP<const VectorBase<Scalar> > colImpl(Ordinal j) const;
  /** \brief . */
  RCP<VectorBase<Scalar> > nonconstColImpl(Ordinal j);

  /** \brief . */
  RCP<const MultiVectorBase<Scalar> >
  contigSubViewImpl(const Range1D& colRng) const;
  /** \brief . */
  RCP<MultiVectorBase<Scalar> >
  nonconstContigSubViewImpl(const Range1D& colRng);
  /** \brief . */
  RCP<const MultiVectorBase<Scalar> >
  nonContigSubViewImpl(const ArrayView<const int>& cols_in) const;
  /** \brief . */
  RCP<MultiVectorBase<Scalar> >
  nonconstNonContigSubViewImpl(const ArrayView<const int>& cols_in);

  /** \brief . */
  virtual void mvMultiReductApplyOpImpl(
    const RTOpPack::RTOpT<Scalar> &primary_op,
    const ArrayView<const Ptr<const MultiVectorBase<Scalar> > > &multi_vecs,
    const ArrayView<const Ptr<MultiVectorBase<Scalar> > > &targ_multi_vecs,
    const ArrayView<const Ptr<RTOpPack::ReductTarget> > &reduct_objs,
    const Ordinal primary_global_offset
    ) const;

  /** \brief . */
  void acquireDetachedMultiVectorViewImpl(
    const Range1D &rowRng,
    const Range1D &colRng,
    RTOpPack::ConstSubMultiVectorView<Scalar>* sub_mv
    ) const;

  /** \brief . */
  void acquireNonconstDetachedMultiVectorViewImpl(
    const Range1D &rowRng,
    const Range1D &colRng,
    RTOpPack::SubMultiVectorView<Scalar>* sub_mv
    );

  /** \brief . */
  void commitNonconstDetachedMultiVectorViewImpl(
    RTOpPack::SubMultiVectorView<Scalar>* sub_mv
    );

//  /** \brief . */
//  RCP<const MultiVectorBase<Scalar> >
//  nonContigSubViewImpl( const ArrayView<const int> &cols ) const;
//  /** \brief . */
//  RCP<MultiVectorBase<Scalar> >
//  nonconstNonContigSubViewImpl( const ArrayView<const int> &cols );
  //@}

  /** @name Overridden protected functions from SpmdMultiVectorBase */
  //@{
  /** \brief . */
  RCP<const SpmdVectorSpaceBase<Scalar> > spmdSpaceImpl() const;
  /** \brief . */
  void getNonconstLocalMultiVectorDataImpl(
    const Ptr<ArrayRCP<Scalar> > &localValues, const Ptr<Ordinal> &leadingDim
    );
  /** \brief . */
  void getLocalMultiVectorDataImpl(
    const Ptr<ArrayRCP<const Scalar> > &localValues, const Ptr<Ordinal> &leadingDim
    ) const;

  //@}

  /** @name Overridden protected functions from MultiVectorAdapterBase */
  //@{
  /** \brief . */
  virtual void euclideanApply(
    const EOpTransp M_trans,
    const MultiVectorBase<Scalar> &X,
    const Ptr<MultiVectorBase<Scalar> > &Y,
    const Scalar alpha,
    const Scalar beta
    ) const;

  //@}

private:

  // ///////////////////////////////////////
  // Private data members

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  RCP<const TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tpetraVectorSpace_;
#else
  RCP<const TpetraVectorSpace<Scalar,Node> > tpetraVectorSpace_;
#endif
  RCP<const ScalarProdVectorSpaceBase<Scalar> > domainSpace_;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  Teuchos::ConstNonconstObjectContainer<Tpetra::MultiVector<Scalar, LocalOrdinal,GlobalOrdinal,Node> >
#else
  Teuchos::ConstNonconstObjectContainer<Tpetra::MultiVector<Scalar,Node> >
#endif
  tpetraMultiVector_;

  // ////////////////////////////////////
  // Private member functions

  template<class TpetraMultiVector_t>
  void initializeImpl(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    const RCP<const TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraVectorSpace,
#else
    const RCP<const TpetraVectorSpace<Scalar,Node> > &tpetraVectorSpace,
#endif
    const RCP<const ScalarProdVectorSpaceBase<Scalar> > &domainSpace,
    const RCP<TpetraMultiVector_t> &tpetraMultiVector
    );

  // Non-throwing Tpetra MultiVector extraction methods.
  // Return null if casting failed.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  RCP<Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
#else
  RCP<Tpetra::MultiVector<Scalar,Node> >
#endif
  getTpetraMultiVector(const RCP<MultiVectorBase<Scalar> >& mv) const;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  RCP<const Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
#else
  RCP<const Tpetra::MultiVector<Scalar,Node> >
#endif
  getConstTpetraMultiVector(const RCP<const MultiVectorBase<Scalar> >& mv) const;

};


/** \brief Nonmember constructor for TpetraMultiVector.
 *
 * \relates TpetraMultiVector.
 */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
#else
template <class Scalar, class Node>
RCP<TpetraMultiVector<Scalar,Node> >
#endif
tpetraMultiVector(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<const TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraVectorSpace,
#else
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
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  RCP<TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tmv =
    Teuchos::rcp(new TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>);
#else
  RCP<TpetraMultiVector<Scalar,Node> > tmv =
    Teuchos::rcp(new TpetraMultiVector<Scalar,Node>);
#endif
  tmv->initialize(tpetraVectorSpace, domainSpace, tpetraMultiVector);
  return tmv;
}


/** \brief Nonmember constructor for TpetraMultiVector.
 *
 * \relates TpetraMultiVector.
 */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
#else
template <class Scalar, class Node>
RCP<const TpetraMultiVector<Scalar,Node> >
#endif
constTpetraMultiVector(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<const TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraVectorSpace,
#else
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
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  RCP<TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tmv =
    Teuchos::rcp(new TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>);
#else
  RCP<TpetraMultiVector<Scalar,Node> > tmv =
    Teuchos::rcp(new TpetraMultiVector<Scalar,Node>);
#endif
  tmv->constInitialize(tpetraVectorSpace, domainSpace, tpetraMultiVector);
  return tmv;
}


} // end namespace Thyra


#endif // THYRA_TPETRA_MULTIVECTOR_DECL_HPP
