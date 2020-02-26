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

#ifndef THYRA_TPETRA_THYRA_WRAPPERS_DECL_HPP
#define THYRA_TPETRA_THYRA_WRAPPERS_DECL_HPP


#include "Thyra_OperatorVectorTypes.hpp"
#include "Thyra_TpetraVectorSpace_decl.hpp"
#include "Thyra_TpetraVector_decl.hpp"
#include "Thyra_TpetraMultiVector_decl.hpp"
#include "Thyra_TpetraLinearOp_decl.hpp"


namespace Thyra {


/** \brief Given an Tpetra <tt>Teuchos::Comm<int></tt> object, return an
 * equivalent <tt>Teuchos::Comm<Ordinal></tt> object.
 *
 * Will throw if conversion is not successful.
 *
 * \ingroup Tpetra_Thyra_Op_Vec_adapters_grp
 */
RCP<const Teuchos::Comm<Ordinal> >
convertTpetraToThyraComm( const RCP<const Teuchos::Comm<int> > &tpetraComm );


/** \brief Create a Thyra::VectorSpaceBase object given a Tpetra::Map.
 *
 * \ingroup Tpetra_Thyra_Op_Vec_adapters_grp
 */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<const VectorSpaceBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
createVectorSpace(const RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > &tpetraMap);
#else
createVectorSpace(const RCP<const Tpetra::Map<Node> > &tpetraMap);
#endif


/** \brief .
 *
 * \ingroup Tpetra_Thyra_Op_Vec_adapters_grp
 */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<VectorBase<Scalar> >
createVector(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<Tpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraVector,
#else
  const RCP<Tpetra::Vector<Scalar,Node> > &tpetraVector,
#endif
  const RCP<const VectorSpaceBase<Scalar> > space = Teuchos::null
  );


/** \brief .
 *
 * \ingroup Tpetra_Thyra_Op_Vec_adapters_grp
 */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<const VectorBase<Scalar> >
createConstVector(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<const Tpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraVector,
#else
  const RCP<const Tpetra::Vector<Scalar,Node> > &tpetraVector,
#endif
  const RCP<const VectorSpaceBase<Scalar> > space = Teuchos::null
  );


/** \brief .
 *
 * \ingroup Tpetra_Thyra_Op_Vec_adapters_grp
 */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<MultiVectorBase<Scalar> >
createMultiVector(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraMultiVector,
#else
  const RCP<Tpetra::MultiVector<Scalar,Node> > &tpetraMultiVector,
#endif
  const RCP<const VectorSpaceBase<Scalar> > rangeSpace = Teuchos::null,
  const RCP<const VectorSpaceBase<Scalar> > domainSpace = Teuchos::null
  );


/** \brief .
 *
 * \ingroup Tpetra_Thyra_Op_Vec_adapters_grp
 */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<const MultiVectorBase<Scalar> >
createConstMultiVector(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<const Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraMultiVector,
#else
  const RCP<const Tpetra::MultiVector<Scalar,Node> > &tpetraMultiVector,
#endif
  const RCP<const VectorSpaceBase<Scalar> > rangeSpace = Teuchos::null,
  const RCP<const VectorSpaceBase<Scalar> > domainSpace = Teuchos::null
  );


/** \brief .
 *
 * \ingroup Tpetra_Thyra_Op_Vec_adapters_grp
 */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<LinearOpBase<Scalar> >
createLinearOp(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<Tpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraOperator,
#else
  const RCP<Tpetra::Operator<Scalar,Node> > &tpetraOperator,
#endif
  const RCP<const VectorSpaceBase<Scalar> > rangeSpace = Teuchos::null,
  const RCP<const VectorSpaceBase<Scalar> > domainSpace = Teuchos::null
  );


/** \brief .
 *
 * \ingroup Tpetra_Thyra_Op_Vec_adapters_grp
 */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template <class Scalar, class Node>
#endif
RCP<const LinearOpBase<Scalar> >
createConstLinearOp(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<const Tpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tpetraOperator,
#else
  const RCP<const Tpetra::Operator<Scalar,Node> > &tpetraOperator,
#endif
  const RCP<const VectorSpaceBase<Scalar> > rangeSpace = Teuchos::null,
  const RCP<const VectorSpaceBase<Scalar> > domainSpace = Teuchos::null
  );


/** \brief Traits class that enables the extraction of Tpetra operator/vector
 * objects wrapped in Thyra operator/vector objects.
 *
 * Example usage:

 \code

  typedef Thyra::TpetraObjectExtraction<Scalar,LO,GO,Node> TOE;
  typedef Tpetra::MultiVector<Scalar,LO,GO,Node> TpetraMultiVector_t;

  RCP<TpetraMultiVector_t> tpetraMv = TOE::getTpetraMultiVector(thyraMv);
  RCP<TpetraVector_t> tpetraV = TOE::getTpetraVector(thyraV);

 \endcode

 *
 * \todo Finish documentation
 *
 * \ingroup Tpetra_Thyra_Op_Vec_adapters_grp
 */
template <class Scalar = Tpetra::Vector<>::scalar_type,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
          class LocalOrdinal = Tpetra::Vector<>::local_ordinal_type,
          class GlobalOrdinal = Tpetra::Vector<>::global_ordinal_type,
#endif
          class Node = Tpetra::Vector<>::node_type>
class TpetraOperatorVectorExtraction {
public:

#ifndef TPETRA_ENABLE_TEMPLATE_ORDINALS
  using LocalOrdinal = typename Tpetra::Map<>::local_ordinal_type;
  using GlobalOrdinal = typename Tpetra::Map<>::global_ordinal_type;
#endif
  /** \brief Get a const Tpetra::Map from a const Thyra::VectorSpaceBase object.
   */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  static RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >
#else
  static RCP<const Tpetra::Map<Node> >
#endif
  getTpetraMap(const RCP<const VectorSpaceBase<Scalar> > &vs);

  /** \brief Get a non-const Tpetra::Vector from a non-const
   * Thyra::VectorBase object.
   */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  static RCP<Tpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
#else
  static RCP<Tpetra::Vector<Scalar,Node> >
#endif
  getTpetraVector(const RCP<VectorBase<Scalar> > &v);

  /** \brief Get a const Tpetra::Vector from a const
   * Thyra::VectorBase object.
   */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  static RCP<const Tpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
#else
  static RCP<const Tpetra::Vector<Scalar,Node> >
#endif
  getConstTpetraVector(const RCP<const VectorBase<Scalar> > &v);

  /** \brief Get a non-const Tpetra::MultiVector from a non-const
   * Thyra::MultiVectorBase object.
   */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  static RCP<Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
#else
  static RCP<Tpetra::MultiVector<Scalar,Node> >
#endif
  getTpetraMultiVector(const RCP<MultiVectorBase<Scalar> > &mv);

  /** \brief Get a const Tpetra::MultiVector from a const
   * Thyra::MultiVectorBase object.
   */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  static RCP<const Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
#else
  static RCP<const Tpetra::MultiVector<Scalar,Node> >
#endif
  getConstTpetraMultiVector(const RCP<const MultiVectorBase<Scalar> > &mv);

  /** \brief Get a non-const Tpetra::Operator from a non-const
   * Thyra::LinearOpBase object.
   */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  static RCP<Tpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
#else
  static RCP<Tpetra::Operator<Scalar,Node> >
#endif
  getTpetraOperator(const RCP<LinearOpBase<Scalar> > &op);

  /** \brief Get a const Tpetra::Operator from a const
   * Thyra::LinearOpBase object.
   */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  static RCP<const Tpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
#else
  static RCP<const Tpetra::Operator<Scalar,Node> >
#endif
  getConstTpetraOperator(const RCP<const LinearOpBase<Scalar> > &op);

};


} // namespace Thyra


#endif // THYRA_TPETRA_THYRA_WRAPPERS_DECL_HPP
