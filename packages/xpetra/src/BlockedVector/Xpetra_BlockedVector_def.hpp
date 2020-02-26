// @HEADER
//
// ***********************************************************************
//
//             Xpetra: A linear algebra interface package
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
//                    Tobias Wiesner    (tawiesn@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef XPETRA_BLOCKEDVECTOR_DEF_HPP
#define XPETRA_BLOCKEDVECTOR_DEF_HPP

#include "Xpetra_BlockedVector_decl.hpp"

#include "Xpetra_BlockedMultiVector.hpp"
#include "Xpetra_Exceptions.hpp"



namespace Xpetra {



#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
BlockedVector(const Teuchos::RCP<const Xpetra::BlockedMap<LocalOrdinal,GlobalOrdinal,Node>>& map, bool zeroOut)
    : Xpetra::BlockedMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>(map, 1, zeroOut)
#else
template<class Scalar, class Node>
BlockedVector<Scalar, Node>::
BlockedVector(const Teuchos::RCP<const Xpetra::BlockedMap<Node>>& map, bool zeroOut)
    : Xpetra::BlockedMultiVector<Scalar, Node>(map, 1, zeroOut)
#endif
{ }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
BlockedVector( Teuchos::RCP<const Xpetra::BlockedMap<LocalOrdinal,GlobalOrdinal,Node>> bmap,
               Teuchos::RCP<Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node>>    v)
    : Xpetra::BlockedMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>(bmap, v)
#else
template<class Scalar, class Node>
BlockedVector<Scalar, Node>::
BlockedVector( Teuchos::RCP<const Xpetra::BlockedMap<Node>> bmap,
               Teuchos::RCP<Xpetra::Vector<Scalar,Node>>    v)
    : Xpetra::BlockedMultiVector<Scalar, Node>(bmap, v)
#endif
{ }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
BlockedVector( Teuchos::RCP<const Xpetra::MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node> > mapExtractor,
               Teuchos::RCP<Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node> > v)
    : Xpetra::BlockedMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>(mapExtractor, v)
#else
template<class Scalar, class Node>
BlockedVector<Scalar, Node>::
BlockedVector( Teuchos::RCP<const Xpetra::MapExtractor<Scalar, Node> > mapExtractor,
               Teuchos::RCP<Xpetra::Vector<Scalar, Node> > v)
    : Xpetra::BlockedMultiVector<Scalar, Node>(mapExtractor, v)
#endif
{ }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Scalar, class Node>
BlockedVector<Scalar, Node>::
#endif
~BlockedVector()
{ }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>&
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
operator=(const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& rhs)
#else
template<class Scalar, class Node>
BlockedVector<Scalar, Node>&
BlockedVector<Scalar, Node>::
operator=(const Xpetra::MultiVector<Scalar, Node>& rhs)
#endif
{
    assign(rhs);      // dispatch to protected virtual method
    return *this;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
replaceGlobalValue(GlobalOrdinal globalRow, size_t vectorIndex, const Scalar& value)
{
    BlockedMultiVector::replaceGlobalValue(globalRow, vectorIndex, value);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
sumIntoGlobalValue(GlobalOrdinal globalRow, size_t vectorIndex, const Scalar& value)
{
    BlockedMultiVector::sumIntoGlobalValue(globalRow, vectorIndex, value);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
replaceLocalValue(LocalOrdinal  myRow, size_t vectorIndex, const Scalar& value)
{
    BlockedMultiVector::replaceLocalValue(myRow, vectorIndex, value);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
sumIntoLocalValue(LocalOrdinal  myRow, size_t vectorIndex, const Scalar& value)
{
    BlockedMultiVector::sumIntoLocalValue(myRow, vectorIndex, value);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
replaceGlobalValue(GlobalOrdinal globalRow, const Scalar& value)
{
    BlockedMultiVector::replaceGlobalValue(globalRow, 0, value);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
sumIntoGlobalValue(GlobalOrdinal globalRow, const Scalar& value)
{
    BlockedMultiVector::sumIntoGlobalValue(globalRow, 0, value);
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
replaceLocalValue(LocalOrdinal myRow, const Scalar& value)
{
    BlockedMultiVector::replaceLocalValue(myRow, 0, value);
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
sumIntoLocalValue(LocalOrdinal myRow, const Scalar& value)
{
    BlockedMultiVector::sumIntoLocalValue(myRow, 0, value);
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
putScalar(const Scalar& value)
{
    BlockedMultiVector::putScalar(value);
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Scalar, class Node>
Teuchos::RCP<const Xpetra::Vector<Scalar, Node>>
BlockedVector<Scalar, Node>::
#endif
getVector(size_t j) const
{
    return BlockedMultiVector::getVector(j);
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Scalar, class Node>
Teuchos::RCP<Xpetra::Vector<Scalar, Node>>
BlockedVector<Scalar, Node>::
#endif
getVectorNonConst(size_t j)
{
    return BlockedMultiVector::getVectorNonConst(j);
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
Teuchos::ArrayRCP<const Scalar>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
getData(size_t j) const
{
    return BlockedMultiVector::getData(j);
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
Teuchos::ArrayRCP<Scalar>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
getDataNonConst(size_t j)
{
    return BlockedMultiVector::getDataNonConst(j);
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
dot(const MultiVector& A, const Teuchos::ArrayView<Scalar>& dots) const
{
    BlockedMultiVector::dot(A, dots);
    return;
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
Scalar
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
dot(const Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& A) const
#else
BlockedVector<Scalar, Node>::
dot(const Xpetra::Vector<Scalar, Node>& A) const
#endif
{
    Teuchos::Array<Scalar> dots = Teuchos::Array<Scalar>(1);
    BlockedMultiVector::dot(A, dots);
    return dots[ 0 ];
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
abs(const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& A)
#else
BlockedVector<Scalar, Node>::
abs(const Xpetra::MultiVector<Scalar, Node>& A)
#endif
{
    BlockedMultiVector::abs(A);
    return;
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
reciprocal(const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& A)
#else
BlockedVector<Scalar, Node>::
reciprocal(const Xpetra::MultiVector<Scalar, Node>& A)
#endif
{
    BlockedMultiVector::reciprocal(A);
    return;
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
scale(const Scalar& alpha)
{
    BlockedMultiVector::scale(alpha);
    return;
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
scale(Teuchos::ArrayView<const Scalar> alpha)
{
    BlockedMultiVector::scale(alpha);
    return;
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
update(const Scalar& alpha,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
       const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& A,
#else
       const Xpetra::MultiVector<Scalar, Node>& A,
#endif
       const Scalar& beta)
{
    BlockedMultiVector::update(alpha, A, beta);
    return;
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
update(const Scalar&                                                         alpha,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
       const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& A,
#else
       const Xpetra::MultiVector<Scalar, Node>& A,
#endif
       const Scalar&                                                         beta,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
       const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& B,
#else
       const Xpetra::MultiVector<Scalar, Node>& B,
#endif
       const Scalar&                                                         gamma)
{
    BlockedMultiVector::update(alpha, A, beta, B, gamma);
    return;
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
typename Teuchos::ScalarTraits<Scalar>::magnitudeType
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
norm1() const
{
    using Array = Teuchos::Array<typename Teuchos::ScalarTraits<Scalar>::magnitudeType>;
    Array norm = Array(1);
    this->norm1(norm);
    return norm[ 0 ];
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
typename Teuchos::ScalarTraits<Scalar>::magnitudeType
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
norm2() const
{
    Teuchos::Array<typename Teuchos::ScalarTraits<Scalar>::magnitudeType> norm =
      Teuchos::Array<typename Teuchos::ScalarTraits<Scalar>::magnitudeType>(1);
    this->norm2(norm);
    return norm[ 0 ];
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
typename Teuchos::ScalarTraits<Scalar>::magnitudeType
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
normInf() const
{
    Teuchos::Array<typename Teuchos::ScalarTraits<Scalar>::magnitudeType>
        norm = Teuchos::Array<typename Teuchos::ScalarTraits<Scalar>::magnitudeType>(1);
    this->normInf(norm);
    return norm[ 0 ];
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
norm1(const Teuchos::ArrayView<typename Teuchos::ScalarTraits<Scalar>::magnitudeType>& norms) const
{
    BlockedMultiVector::norm1(norms);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
norm2(const Teuchos::ArrayView<typename Teuchos::ScalarTraits<Scalar>::magnitudeType>& norms) const
{
    BlockedMultiVector::norm2(norms);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
normInf(const Teuchos::ArrayView<typename Teuchos::ScalarTraits<Scalar>::magnitudeType>& norms) const
{
    BlockedMultiVector::normInf(norms);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
meanValue(const Teuchos::ArrayView<Scalar>& /* means */) const
{
    throw Xpetra::Exceptions::RuntimeError("BlockedVector::meanValue: Not (yet) supported by BlockedVector.");
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
Scalar
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
meanValue() const
{
    throw Xpetra::Exceptions::RuntimeError("BlockedVector::meanValue: Not (yet) supported by BlockedVector.");
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
multiply(Teuchos::ETransp /* transA */,
         Teuchos::ETransp /* transB */,
         const Scalar&    /* alpha */,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
         const Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>&    /* A */,
         const Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>&    /* B */,
#else
         const Xpetra::Vector<Scalar, Node>&    /* A */,
         const Xpetra::Vector<Scalar, Node>&    /* B */,
#endif
         const Scalar&    /* beta */)
{
    throw Xpetra::Exceptions::RuntimeError("BlockedVector::multiply: Not (yet) supported by BlockedVector.");
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
multiply(Teuchos::ETransp   /* transA */,
         Teuchos::ETransp   /* transB */,
         const Scalar&      /* alpha */,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
         const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& /* A */,
         const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& /* B */,
#else
         const Xpetra::MultiVector<Scalar, Node>& /* A */,
         const Xpetra::MultiVector<Scalar, Node>& /* B */,
#endif
         const Scalar&      /* beta */)
{
    throw Xpetra::Exceptions::RuntimeError("BlockedVector::multiply: Not (yet) supported by BlockedVector.");
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
elementWiseMultiply( Scalar /* scalarAB */,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                     const Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& /* A */,
                     const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& /* B */,
#else
                     const Xpetra::Vector<Scalar, Node>& /* A */,
                     const Xpetra::MultiVector<Scalar, Node>& /* B */,
#endif
                     Scalar /* scalarThis */)
{
    throw Xpetra::Exceptions::RuntimeError("BlockedVector::elementWiseMultiply: Not (yet) supported by BlockedVector.");
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
elementWiseMultiply( Scalar /* scalarAB */,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                     const Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& A,
                     const Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& B,
#else
                     const Xpetra::Vector<Scalar, Node>& A,
                     const Xpetra::Vector<Scalar, Node>& B,
#endif
                     Scalar /* scalarThis */)
{
    XPETRA_TEST_FOR_EXCEPTION(B.getMap()->isSameAs(*(this->getMap())) == false,
                              Xpetra::Exceptions::RuntimeError,
                              "BlockedVector::elementWiseMultipy: B must have same blocked map than this.");
    TEUCHOS_TEST_FOR_EXCEPTION(A.getMap()->getNodeNumElements() != B.getMap()->getNodeNumElements(),
                               Xpetra::Exceptions::RuntimeError,
                               "BlockedVector::elementWiseMultipy: A has "
                                 << A.getMap()->getNodeNumElements() << " elements, B has " << B.getMap()->getNodeNumElements()
                                 << ".");
    TEUCHOS_TEST_FOR_EXCEPTION(A.getMap()->getGlobalNumElements() != B.getMap()->getGlobalNumElements(),
                               Xpetra::Exceptions::RuntimeError,
                               "BlockedVector::elementWiseMultipy: A has " << A.getMap()->getGlobalNumElements()
                                                                           << " elements, B has "
                                                                           << B.getMap()->getGlobalNumElements() << ".");

    RCP<const BlockedMap>                                                bmap  = this->getBlockedMap();
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<const Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> rcpA  = Teuchos::rcpFromRef(A);
    RCP<const Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> bmvec = Teuchos::rcpFromRef(B);
#else
    RCP<const Xpetra::Vector<Scalar, Node>> rcpA  = Teuchos::rcpFromRef(A);
    RCP<const Xpetra::Vector<Scalar, Node>> bmvec = Teuchos::rcpFromRef(B);
#endif
    RCP<const BlockedVector> bbmvec = Teuchos::rcp_dynamic_cast<const BlockedVector>(bmvec);
    TEUCHOS_TEST_FOR_EXCEPTION(bbmvec.is_null() == true,
                               Xpetra::Exceptions::RuntimeError,
                               "BlockedVector::elementWiseMultipy: B must be a BlockedVector.");

    // TODO implement me
    /*RCP<Xpetra::MapExtractor<Scalar,LocalOrdinal,GlobalOrdinal,Node> > me = Teuchos::rcp(new
    Xpetra::MapExtractor<Scalar,LocalOrdinal,GlobalOrdinal,Node>(bmap));

    for(size_t m = 0; m < bmap->getNumMaps(); m++) {
      // TODO introduce BlockedVector objects and "skip" this expensive ExtractVector call
      RCP<const Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > pd = me->ExtractVector(rcpA,m,bmap->getThyraMode());
      XPETRA_TEST_FOR_EXCEPTION(pd->getMap()->isSameAs(*(this->getBlockedMap()->getMap(m,bmap->getThyraMode())))==false,
    Xpetra::Exceptions::RuntimeError, "BlockedVector::elementWiseMultipy: sub map of B does not fit with sub map of this.");
      this->getMultiVector(m,bmap->getThyraMode())->elementWiseMultiply(scalarAB,*pd,*(bbmvec->getMultiVector(m,bmap->getThyraMode())),scalarThis);
    }*/
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
size_t
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
getNumVectors() const
{
    return 1;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
size_t
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
getLocalLength() const
{
    throw Xpetra::Exceptions::RuntimeError(
      "BlockedVector::getLocalLength: routine not implemented. It has no value as one must iterate on the partial vectors.");
    TEUCHOS_UNREACHABLE_RETURN(0);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
global_size_t
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
getGlobalLength() const
{
    return this->getBlockedMap()->getFullMap()->getGlobalNumElements();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
bool
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
isSameSize(const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& /* vec */) const
#else
BlockedVector<Scalar, Node>::
isSameSize(const Xpetra::MultiVector<Scalar, Node>& /* vec */) const
#endif
{
    throw Xpetra::Exceptions::RuntimeError(
      "BlockedVector::isSameSize: routine not implemented. It has no value as one must iterate on the partial vectors.");
    TEUCHOS_UNREACHABLE_RETURN(0);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
std::string
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
description() const
{
    return std::string("BlockedVector");
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
describe(Teuchos::FancyOStream& out, const Teuchos::EVerbosityLevel verbLevel) const
{
    out << description() << std::endl;
    for(size_t r = 0; r < this->getBlockedMap()->getNumMaps(); r++)
    {
        getMultiVector(r)->describe(out, verbLevel);
    }
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
replaceMap(const RCP<const Map>& map)
{
    BlockedMultiVector::replaceMap(map);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
doImport(const DistObject<Scalar, LocalOrdinal, GlobalOrdinal, Node>& /* source */,
#else
BlockedVector<Scalar, Node>::
doImport(const DistObject<Scalar, Node>& /* source */,
#endif
         const Import& /* importer */,
         CombineMode /* CM */)
{
    throw Xpetra::Exceptions::RuntimeError("BlockedVector::doImport: Not supported by BlockedVector.");
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
doExport(const DistObject<Scalar, LocalOrdinal, GlobalOrdinal, Node>& /* dest */,
#else
BlockedVector<Scalar, Node>::
doExport(const DistObject<Scalar, Node>& /* dest */,
#endif
         const Import& /* importer */,
         CombineMode /* CM */)
{
    throw Xpetra::Exceptions::RuntimeError("BlockedVector::doExport: Not supported by BlockedVector.");
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
doImport(const DistObject<Scalar, LocalOrdinal, GlobalOrdinal, Node>& /* source */,
#else
BlockedVector<Scalar, Node>::
doImport(const DistObject<Scalar, Node>& /* source */,
#endif
         const Export& /* exporter */,
         CombineMode /* CM */)
{
    throw Xpetra::Exceptions::RuntimeError("BlockedVector::doImport: Not supported by BlockedVector.");
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
doExport(const DistObject<Scalar, LocalOrdinal, GlobalOrdinal, Node>& /* dest */,
#else
BlockedVector<Scalar, Node>::
doExport(const DistObject<Scalar, Node>& /* dest */,
#endif
         const Export& /* exporter */,
         CombineMode /* CM */)
{
    throw Xpetra::Exceptions::RuntimeError("BlockedVector::doExport: Not supported by BlockedVector.");
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
setSeed(unsigned int seed)
{
    for(size_t r = 0; r < this->getBlockedMap()->getNumMaps(); ++r)
    {
        getMultiVector(r)->setSeed(seed);
    }
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
randomize(bool bUseXpetraImplementation)
{
    for(size_t r = 0; r < this->getBlockedMap()->getNumMaps(); ++r)
    {
        getMultiVector(r)->randomize(bUseXpetraImplementation);
    }
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
BlockedVector<Scalar, Node>::
#endif
Xpetra_randomize()
{
    {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Xpetra_randomize();
#else
        Xpetra::Vector<Scalar, Node>::Xpetra_randomize();
#endif
    }
}

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR

#if 0
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    template<class TargetDeviceType>
    typename Kokkos::Impl::if_c<Kokkos::Impl::is_same<typename dev_execution_space::memory_space,
                                                      typename TargetDeviceType::memory_space>::value,
                                typename dual_view_type::t_dev_um,
                                typename dual_view_type::t_host_um>::type
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    BlockedVector<Scalar, Node>::
#endif
    getLocalView() const
    {
        if(Kokkos::Impl::is_same<typename host_execution_space::memory_space, typename TargetDeviceType::memory_space>::value)
        {
            return getHostLocalView();
        }
        else
        {
            return getDeviceLocalView();
        }
    }
#endif

//    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
//    virtual typename dual_view_type::
//    t_dev_um BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getDeviceLocalView() const
//    {
//        typename dual_view_type::t_dev_um test;
//        return test;
//    }
#endif      // HAVE_XPETRA_KOKKOS_REFACTOR


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    Teuchos::RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >
    BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    template<class Scalar, class Node>
    Teuchos::RCP<const Xpetra::Map<Node> >
    BlockedVector<Scalar, Node>::
#endif
    getMap() const
    {
        XPETRA_MONITOR("BlockedVector::getMap");
        return this->getBlockedMap();
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    Teuchos::RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
    BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    template<class Scalar, class Node>
    Teuchos::RCP<Xpetra::MultiVector<Scalar, Node> >
    BlockedVector<Scalar, Node>::
#endif
    getMultiVector(size_t r) const
    {
        return BlockedMultiVector::getMultiVector(r);
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    Teuchos::RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
    BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    template<class Scalar, class Node>
    Teuchos::RCP<Xpetra::MultiVector<Scalar, Node> >
    BlockedVector<Scalar, Node>::
#endif
    getMultiVector(size_t r, bool bThyraMode) const
    {
        return BlockedMultiVector::getMultiVector(r, bThyraMode);
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    BlockedVector<Scalar, Node>::
#endif
    setMultiVector(size_t r,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                   Teuchos::RCP<const Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node> > v,
#else
                   Teuchos::RCP<const Xpetra::Vector<Scalar, Node> > v,
#endif
                   bool bThyraMode)
    {
        BlockedMultiVector::setMultiVector(r, v, bThyraMode);
        return;
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    Teuchos::RCP< Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
    BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    template<class Scalar, class Node>
    Teuchos::RCP< Xpetra::MultiVector<Scalar, Node> >
    BlockedVector<Scalar, Node>::
#endif
    Merge() const
    {
        return BlockedMultiVector::Merge();
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    BlockedVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    assign(const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& rhs)
#else
    BlockedVector<Scalar, Node>::
    assign(const Xpetra::MultiVector<Scalar, Node>& rhs)
#endif
    {
        BlockedMultiVector::assign(rhs);
    }


    // template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    // virtual void BlockedVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
    // assign (const XpetrA::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& rhs)
    // {
    //     throw Xpetra::Exceptions::RuntimeError("BlockedVector::assign: Not supported by BlockedVector.");
    // }

}      // Xpetra namespace


#endif      // XPETRA_BLOCKEDVECTOR_DEF_HPP
