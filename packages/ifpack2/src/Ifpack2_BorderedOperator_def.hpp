/*@HEADER
// ***********************************************************************
//
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//                 Copyright (2009) Sandia Corporation
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
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
//@HEADER
*/


#ifndef IFPACK2_BORDEREDOPERATOR_DEF_HPP
#define IFPACK2_BORDEREDOPERATOR_DEF_HPP

#include "Ifpack2_BorderedOperator_decl.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Teuchos_TestForException.hpp"

namespace Ifpack2 {

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template< class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node >
BorderedOperator<Scalar, LocalOrdinal, GlobalOrdinal, Node >::
BorderedOperator (const Teuchos::RCP<const Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node > >& A) : 
#else
template< class Scalar, class Node >
BorderedOperator<Scalar, Node >::
BorderedOperator (const Teuchos::RCP<const Tpetra::Operator<Scalar, Node > >& A) : 
#endif
  A_ (A)
{ 
  TEUCHOS_TEST_FOR_EXCEPTION(
    A_.is_null (), std::runtime_error, 
    Teuchos::typeName (*this) << "::BorderedOperator constructor: "
    "The input Operator A is null.");
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template< class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node >
Teuchos::RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >
BorderedOperator<Scalar, LocalOrdinal, GlobalOrdinal, Node >::getDomainMap() const
#else
template< class Scalar, class Node >
Teuchos::RCP<const Tpetra::Map<Node> >
BorderedOperator<Scalar, Node >::getDomainMap() const
#endif
{
  return A_->getDomainMap();
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template< class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node >
Teuchos::RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >
BorderedOperator<Scalar, LocalOrdinal, GlobalOrdinal, Node >::getRangeMap() const
#else
template< class Scalar, class Node >
Teuchos::RCP<const Tpetra::Map<Node> >
BorderedOperator<Scalar, Node >::getRangeMap() const
#endif
{
  return A_->getRangeMap();
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template< class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node >
#else
template< class Scalar, class Node >
#endif
bool
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BorderedOperator<Scalar, LocalOrdinal, GlobalOrdinal, Node >::hasTransposeApply() const 
#else
BorderedOperator<Scalar, Node >::hasTransposeApply() const 
#endif
{
  return A_->hasTransposeApply();
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template< class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node >
#else
template< class Scalar, class Node >
#endif
void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BorderedOperator<Scalar, LocalOrdinal, GlobalOrdinal, Node >::
apply (const Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node >& X,
       Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node >& Y,
#else
BorderedOperator<Scalar, Node >::
apply (const Tpetra::MultiVector<Scalar, Node >& X,
       Tpetra::MultiVector<Scalar, Node >& Y,
#endif
       Teuchos::ETransp mode, 
       Scalar coefAx, 
       Scalar coefY ) const 
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    X.getNumVectors() != Y.getNumVectors(), std::runtime_error,
    "Ifpack2::BorderedOperator::apply(): X.getNumVectors() = " 
    << X.getNumVectors() << " != Y.getNumVectors() = " 
    << Y.getNumVectors() << ".");
  A_->apply (X, Y, mode, coefAx, coefY );
}

} // namespace Ifpack2

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
#define IFPACK2_BORDEREDOPERATOR_INSTANT(S,LO,GO,N) \
  template class Ifpack2::BorderedOperator< S, LO, GO, N >;
#else
#define IFPACK2_BORDEREDOPERATOR_INSTANT(S,N) \
  template class Ifpack2::BorderedOperator< S, N >;
#endif

#endif /* IFPACK2_BorderedOperator_DEF_HPP */
