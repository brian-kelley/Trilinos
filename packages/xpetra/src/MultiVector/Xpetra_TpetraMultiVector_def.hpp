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
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef XPETRA_TPETRAMULTIVECTOR_DEF_HPP
#define XPETRA_TPETRAMULTIVECTOR_DEF_HPP
#include "Xpetra_TpetraConfigDefs.hpp"

#include "Xpetra_TpetraMap.hpp" //TMP
#include "Xpetra_Utils.hpp"
#include "Xpetra_TpetraImport.hpp"
#include "Xpetra_TpetraExport.hpp"

#include "Xpetra_TpetraMultiVector_decl.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Vector.hpp"

namespace Xpetra {


  //! Basic constuctor.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraMultiVector(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &map, size_t NumVectors, bool zeroOut)
    : vec_(Teuchos::rcp(new Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node >(toTpetra(map), NumVectors, zeroOut))) {
#else
  template<class Scalar, class Node>
  TpetraMultiVector<Scalar,Node>::TpetraMultiVector(const Teuchos::RCP< const Map<Node > > &map, size_t NumVectors, bool zeroOut)
    : vec_(Teuchos::rcp(new Tpetra::MultiVector< Scalar, Node >(toTpetra(map), NumVectors, zeroOut))) {
#endif
    // TAW 1/30/2016: even though Tpetra allows numVecs == 0, Epetra does not. Introduce exception to keep behavior of Epetra and Tpetra consistent.
    TEUCHOS_TEST_FOR_EXCEPTION(NumVectors < 1, std::invalid_argument, "Xpetra::TpetraMultiVector(map,numVecs,zeroOut): numVecs = " << NumVectors << " < 1.");
    }
  
  //! Copy constructor (performs a deep copy).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::  
  TpetraMultiVector(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &source)
      : vec_(Teuchos::rcp(new Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node >(Tpetra::createCopy(toTpetra(source))))) {  }
#else
  template<class Scalar, class Node>
  TpetraMultiVector<Scalar,Node>::  
  TpetraMultiVector(const MultiVector< Scalar, Node > &source)
      : vec_(Teuchos::rcp(new Tpetra::MultiVector< Scalar, Node >(Tpetra::createCopy(toTpetra(source))))) {  }
#endif

  //! Create multivector by copying two-dimensional array of local data.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  TpetraMultiVector(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &map, const Teuchos::ArrayView< const Scalar > &A, size_t LDA, size_t NumVectors)
      : vec_(Teuchos::rcp(new Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node >(toTpetra(map), A, LDA, NumVectors))) {
#else
  template<class Scalar, class Node>
  TpetraMultiVector<Scalar,Node>::
  TpetraMultiVector(const Teuchos::RCP< const Map<Node > > &map, const Teuchos::ArrayView< const Scalar > &A, size_t LDA, size_t NumVectors)
      : vec_(Teuchos::rcp(new Tpetra::MultiVector< Scalar, Node >(toTpetra(map), A, LDA, NumVectors))) {
#endif
      // TAW 1/30/2016: even though Tpetra allows numVecs == 0, Epetra does not. Introduce exception to keep behavior of Epetra and Tpetra consistent.
      TEUCHOS_TEST_FOR_EXCEPTION(NumVectors < 1, std::invalid_argument, "Xpetra::TpetraMultiVector(map,A,LDA,numVecs): numVecs = " << NumVectors << " < 1.");
  }

  //! Create multivector by copying array of views of local data.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  TpetraMultiVector(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &map, const Teuchos::ArrayView< const Teuchos::ArrayView< const Scalar > > &ArrayOfPtrs, size_t NumVectors)
      : vec_(Teuchos::rcp(new Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node >(toTpetra(map), ArrayOfPtrs, NumVectors))) {
#else
  template<class Scalar, class Node>
  TpetraMultiVector<Scalar,Node>::
  TpetraMultiVector(const Teuchos::RCP< const Map<Node > > &map, const Teuchos::ArrayView< const Teuchos::ArrayView< const Scalar > > &ArrayOfPtrs, size_t NumVectors)
      : vec_(Teuchos::rcp(new Tpetra::MultiVector< Scalar, Node >(toTpetra(map), ArrayOfPtrs, NumVectors))) {
#endif
      // TAW 1/30/2016: even though Tpetra allows numVecs == 0, Epetra does not. Introduce exception to keep behavior of Epetra and Tpetra consistent.
      TEUCHOS_TEST_FOR_EXCEPTION(NumVectors < 1, std::invalid_argument, "Xpetra::TpetraMultiVector(map,ArrayOfPtrs,numVecs): numVecs = " << NumVectors << " < 1.");
  }


  //! Destructor (virtual for memory safety of derived classes).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  template<class Scalar, class Node>
  TpetraMultiVector<Scalar,Node>::
#endif
  ~TpetraMultiVector() {  }

  //! Replace value, using global (row) index.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::
#endif
  replaceGlobalValue(GlobalOrdinal globalRow, size_t vectorIndex, const Scalar &value) { XPETRA_MONITOR("TpetraMultiVector::replaceGlobalValue"); vec_->replaceGlobalValue(globalRow, vectorIndex, value); }

  //! Add value to existing value, using global (row) index.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::
#endif
  sumIntoGlobalValue(GlobalOrdinal globalRow, size_t vectorIndex, const Scalar &value) { XPETRA_MONITOR("TpetraMultiVector::sumIntoGlobalValue"); vec_->sumIntoGlobalValue(globalRow, vectorIndex, value); }

  //! Replace value, using local (row) index.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::
#endif
  replaceLocalValue(LocalOrdinal myRow, size_t vectorIndex, const Scalar &value) { XPETRA_MONITOR("TpetraMultiVector::replaceLocalValue"); vec_->replaceLocalValue(myRow, vectorIndex, value); }

  //! Add value to existing value, using local (row) index
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::
#endif
  sumIntoLocalValue(LocalOrdinal myRow, size_t vectorIndex, const Scalar &value) { XPETRA_MONITOR("TpetraMultiVector::sumIntoLocalValue"); vec_->sumIntoLocalValue(myRow, vectorIndex, value); }

  //! Set all values in the multivector with the given value
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::
#endif
  putScalar(const Scalar &value) { XPETRA_MONITOR("TpetraMultiVector::putScalar"); vec_->putScalar(value); }

  //! Sum values of a locally replicated multivector across all processes.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::
#endif
  reduce() { XPETRA_MONITOR("TpetraMultiVector::reduce"); vec_->reduce(); }

  
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  Teuchos::RCP< const Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > > 
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  template<class Scalar, class Node>
  Teuchos::RCP< const Vector< Scalar, Node > > 
  TpetraMultiVector<Scalar,Node>::
#endif
  getVector(size_t j) const { XPETRA_MONITOR("TpetraMultiVector::getVector"); return toXpetra(vec_->getVector(j)); }

  //! Return a Vector which is a nonconst view of column j.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  Teuchos::RCP< Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > > 
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  template<class Scalar, class Node>
  Teuchos::RCP< Vector< Scalar, Node > > 
  TpetraMultiVector<Scalar,Node>::
#endif
  getVectorNonConst(size_t j) { XPETRA_MONITOR("TpetraMultiVector::getVectorNonConst"); return toXpetra(vec_->getVectorNonConst(j)); }

  //! Const view of the local values in a particular vector of this multivector.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
  template<class Scalar, class Node>
#endif
  Teuchos::ArrayRCP< const Scalar > 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  TpetraMultiVector<Scalar,Node>::
#endif
  getData(size_t j) const { XPETRA_MONITOR("TpetraMultiVector::getData"); return vec_->getData(j); }

  //! View of the local values in a particular vector of this multivector.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
  template<class Scalar, class Node>
#endif
  Teuchos::ArrayRCP< Scalar > 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  TpetraMultiVector<Scalar,Node>::
#endif
  getDataNonConst(size_t j) { XPETRA_MONITOR("TpetraMultiVector::getDataNonConst"); return vec_->getDataNonConst(j); }
  
  //! Fill the given array with a copy of this multivector's local values.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
  template<class Scalar, class Node>
#endif
  void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  TpetraMultiVector<Scalar,Node>::
#endif
  get1dCopy(Teuchos::ArrayView< Scalar > A, size_t LDA) const { XPETRA_MONITOR("TpetraMultiVector::get1dCopy"); vec_->get1dCopy(A, LDA); }

  //! Fill the given array with a copy of this multivector's local values.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
  template<class Scalar, class Node>
#endif
  void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  TpetraMultiVector<Scalar,Node>::
#endif
  get2dCopy(Teuchos::ArrayView< const Teuchos::ArrayView< Scalar > > ArrayOfPtrs) const { XPETRA_MONITOR("TpetraMultiVector::get2dCopy"); vec_->get2dCopy(ArrayOfPtrs); }

  //! Const persisting (1-D) view of this multivector's local values.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
  template<class Scalar, class Node>
#endif
  Teuchos::ArrayRCP< const Scalar > 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  TpetraMultiVector<Scalar,Node>::
#endif
  get1dView() const { XPETRA_MONITOR("TpetraMultiVector::get1dView"); return vec_->get1dView(); }

  //! Return const persisting pointers to values.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
  template<class Scalar, class Node>
#endif
  Teuchos::ArrayRCP< Teuchos::ArrayRCP< const Scalar > > 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  TpetraMultiVector<Scalar,Node>::
#endif
  get2dView() const { XPETRA_MONITOR("TpetraMultiVector::get2dView"); return vec_->get2dView(); }

  //! Nonconst persisting (1-D) view of this multivector's local values.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
  template<class Scalar, class Node>
#endif
  Teuchos::ArrayRCP< Scalar > 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  TpetraMultiVector<Scalar,Node>::
#endif
  get1dViewNonConst() { XPETRA_MONITOR("TpetraMultiVector::get1dViewNonConst"); return vec_->get1dViewNonConst(); }

  //! Return non-const persisting pointers to values.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
  template<class Scalar, class Node>
#endif
  Teuchos::ArrayRCP< Teuchos::ArrayRCP< Scalar > > 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  TpetraMultiVector<Scalar,Node>::
#endif
  get2dViewNonConst() { XPETRA_MONITOR("TpetraMultiVector::get2dViewNonConst"); return vec_->get2dViewNonConst(); }

  //! Compute dot product of each corresponding pair of vectors, dots[i] = this[i].dot(A[i])
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  dot(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const Teuchos::ArrayView< Scalar > &dots) const { XPETRA_MONITOR("TpetraMultiVector::dot"); vec_->dot(toTpetra(A), dots); }
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::
  dot(const MultiVector< Scalar, Node > &A, const Teuchos::ArrayView< Scalar > &dots) const { XPETRA_MONITOR("TpetraMultiVector::dot"); vec_->dot(toTpetra(A), dots); }
#endif

  //! Put element-wise absolute values of input Multi-vector in target: A = abs(this).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  abs(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A) { XPETRA_MONITOR("TpetraMultiVector::abs"); vec_->abs(toTpetra(A)); }
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::
  abs(const MultiVector< Scalar, Node > &A) { XPETRA_MONITOR("TpetraMultiVector::abs"); vec_->abs(toTpetra(A)); }
#endif

  //! Put element-wise reciprocal values of input Multi-vector in target, this(i,j) = 1/A(i,j).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  reciprocal(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A) { XPETRA_MONITOR("TpetraMultiVector::reciprocal"); vec_->reciprocal(toTpetra(A)); }
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::
  reciprocal(const MultiVector< Scalar, Node > &A) { XPETRA_MONITOR("TpetraMultiVector::reciprocal"); vec_->reciprocal(toTpetra(A)); }
#endif

  //! Scale the current values of a multi-vector, this = alpha*this.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::  
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::  
#endif
  scale(const Scalar &alpha) { XPETRA_MONITOR("TpetraMultiVector::scale"); vec_->scale(alpha); }

  //! Scale the current values of a multi-vector, this[j] = alpha[j]*this[j].
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::
#endif
  scale(Teuchos::ArrayView< const Scalar > alpha) { XPETRA_MONITOR("TpetraMultiVector::scale"); vec_->scale(alpha); }

  //! Replace multi-vector values with scaled values of A, this = alpha*A.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  scale(const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A) { XPETRA_MONITOR("TpetraMultiVector::scale"); vec_->scale(alpha, toTpetra(A)); }
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::
  scale(const Scalar &alpha, const MultiVector< Scalar, Node > &A) { XPETRA_MONITOR("TpetraMultiVector::scale"); vec_->scale(alpha, toTpetra(A)); }
#endif

  //! Update multi-vector values with scaled values of A, this = beta*this + alpha*A.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  update(const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const Scalar &beta) { XPETRA_MONITOR("TpetraMultiVector::update"); vec_->update(alpha, toTpetra(A), beta); }
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::
  update(const Scalar &alpha, const MultiVector< Scalar, Node > &A, const Scalar &beta) { XPETRA_MONITOR("TpetraMultiVector::update"); vec_->update(alpha, toTpetra(A), beta); }
#endif

  //! Update multi-vector with scaled values of A and B, this = gamma*this + alpha*A + beta*B.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  update(const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const Scalar &beta, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &B, const Scalar &gamma) { XPETRA_MONITOR("TpetraMultiVector::update"); vec_->update(alpha, toTpetra(A), beta, toTpetra(B), gamma); }
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::
  update(const Scalar &alpha, const MultiVector< Scalar, Node > &A, const Scalar &beta, const MultiVector< Scalar, Node > &B, const Scalar &gamma) { XPETRA_MONITOR("TpetraMultiVector::update"); vec_->update(alpha, toTpetra(A), beta, toTpetra(B), gamma); }
#endif

  //! Compute 1-norm of each vector in multi-vector.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::
#endif
  norm1(const Teuchos::ArrayView< typename Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const { XPETRA_MONITOR("TpetraMultiVector::norm1"); vec_->norm1(norms); }

    //!
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::  
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::  
#endif
  norm2(const Teuchos::ArrayView< typename Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const { XPETRA_MONITOR("TpetraMultiVector::norm2"); vec_->norm2(norms); }

  //! Compute Inf-norm of each vector in multi-vector.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::
#endif
  normInf(const Teuchos::ArrayView< typename Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const { XPETRA_MONITOR("TpetraMultiVector::normInf"); vec_->normInf(norms); }

  //! Compute mean (average) value of each vector in multi-vector. The outcome of this routine is undefined for non-floating point scalar types (e.g., int).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::
#endif
  meanValue(const Teuchos::ArrayView< Scalar > &means) const { XPETRA_MONITOR("TpetraMultiVector::meanValue"); vec_->meanValue(means); }

  //! Matrix-matrix multiplication: this = beta*this + alpha*op(A)*op(B).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
  multiply(Teuchos::ETransp transA, Teuchos::ETransp transB, const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &B, const Scalar &beta) { XPETRA_MONITOR("TpetraMultiVector::multiply"); vec_->multiply(transA, transB, alpha, toTpetra(A), toTpetra(B), beta); }
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::    
  multiply(Teuchos::ETransp transA, Teuchos::ETransp transB, const Scalar &alpha, const MultiVector< Scalar, Node > &A, const MultiVector< Scalar, Node > &B, const Scalar &beta) { XPETRA_MONITOR("TpetraMultiVector::multiply"); vec_->multiply(transA, transB, alpha, toTpetra(A), toTpetra(B), beta); }
#endif


  //! Number of columns in the multivector.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  size_t   TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
#else
  template<class Scalar, class Node>
  size_t   TpetraMultiVector<Scalar,Node>::    
#endif
  getNumVectors() const { XPETRA_MONITOR("TpetraMultiVector::getNumVectors"); return vec_->getNumVectors(); }

  //! Local number of rows on the calling process.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  size_t TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
#else
  template<class Scalar, class Node>
  size_t TpetraMultiVector<Scalar,Node>::    
#endif
  getLocalLength() const { XPETRA_MONITOR("TpetraMultiVector::getLocalLength"); return vec_->getLocalLength(); }

  //! Global number of rows in the multivector.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  global_size_t TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
#else
  template<class Scalar, class Node>
  global_size_t TpetraMultiVector<Scalar,Node>::    
#endif
  getGlobalLength() const { XPETRA_MONITOR("TpetraMultiVector::getGlobalLength"); return vec_->getGlobalLength(); }

    // \brief Checks to see if the local length, number of vectors and size of Scalar type match
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  bool TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
  isSameSize(const MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> & vec) const { XPETRA_MONITOR("TpetraMultiVector::isSameSize"); return vec_->isSameSize(toTpetra(vec));}
#else
  template<class Scalar, class Node>
  bool TpetraMultiVector<Scalar,Node>::    
  isSameSize(const MultiVector<Scalar,Node> & vec) const { XPETRA_MONITOR("TpetraMultiVector::isSameSize"); return vec_->isSameSize(toTpetra(vec));}
#endif
                          
  //! A simple one-line description of this object.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  std::string TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
#else
  template<class Scalar, class Node>
  std::string TpetraMultiVector<Scalar,Node>::    
#endif
  description() const { XPETRA_MONITOR("TpetraMultiVector::description"); return vec_->description(); }

  //! Print the object with the given verbosity level to a FancyOStream.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::    
#endif
  describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel) const { XPETRA_MONITOR("TpetraMultiVector::describe"); vec_->describe(out, verbLevel); }

  //! Set multi-vector values to random numbers.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::    
#endif
  randomize(bool bUseXpetraImplementation) {
    XPETRA_MONITOR("TpetraMultiVector::randomize");
    
    if(bUseXpetraImplementation)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node >::Xpetra_randomize();
#else
      MultiVector< Scalar, Node >::Xpetra_randomize();
#endif
    else
      vec_->randomize();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  Teuchos::RCP< const Map<LocalOrdinal,GlobalOrdinal,Node> > 
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
#else
  template<class Scalar, class Node>
  Teuchos::RCP< const Map<Node> > 
  TpetraMultiVector<Scalar,Node>::    
#endif
  getMap() const { XPETRA_MONITOR("TpetraMultiVector::getMap"); return toXpetra(vec_->getMap()); }
  
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
  doImport(const DistObject< Scalar, LocalOrdinal,GlobalOrdinal,Node> &source, const Import<LocalOrdinal,GlobalOrdinal,Node> &importer, CombineMode CM) {
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::    
  doImport(const DistObject< Scalar,Node> &source, const Import<Node> &importer, CombineMode CM) {
#endif
    XPETRA_MONITOR("TpetraMultiVector::doImport");
    
    XPETRA_DYNAMIC_CAST(const TpetraMultiVectorClass, source, tSource, "Xpetra::TpetraMultiVector::doImport only accept Xpetra::TpetraMultiVector as input arguments."); //TODO: remove and use toTpetra()
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal,Node> > v = tSource.getTpetra_MultiVector();
#else
    RCP< const Tpetra::MultiVector< Scalar,Node> > v = tSource.getTpetra_MultiVector();
#endif
    this->getTpetra_MultiVector()->doImport(*v, toTpetra(importer), toTpetra(CM));
  }
  
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>  
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
  doExport(const DistObject< Scalar, LocalOrdinal, GlobalOrdinal, Node > &dest, const Import<LocalOrdinal,GlobalOrdinal,Node>& importer, CombineMode CM) {
#else
  template<class Scalar, class Node>  
  void TpetraMultiVector<Scalar,Node>::    
  doExport(const DistObject< Scalar, Node > &dest, const Import<Node>& importer, CombineMode CM) {
#endif
    XPETRA_MONITOR("TpetraMultiVector::doExport");
    
    XPETRA_DYNAMIC_CAST(const TpetraMultiVectorClass, dest, tDest, "Xpetra::TpetraMultiVector::doImport only accept Xpetra::TpetraMultiVector as input arguments."); //TODO: remove and use toTpetra()
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal,Node> > v = tDest.getTpetra_MultiVector();
#else
    RCP< const Tpetra::MultiVector< Scalar,Node> > v = tDest.getTpetra_MultiVector();
#endif
    this->getTpetra_MultiVector()->doExport(*v, toTpetra(importer), toTpetra(CM));
    
  }
  
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
  doImport(const DistObject< Scalar, LocalOrdinal, GlobalOrdinal, Node > &source, const Export<LocalOrdinal,GlobalOrdinal,Node>& exporter, CombineMode CM) {
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::    
  doImport(const DistObject< Scalar, Node > &source, const Export<Node>& exporter, CombineMode CM) {
#endif
    XPETRA_MONITOR("TpetraMultiVector::doImport");
    
    XPETRA_DYNAMIC_CAST(const TpetraMultiVectorClass, source, tSource, "Xpetra::TpetraMultiVector::doImport only accept Xpetra::TpetraMultiVector as input arguments."); //TODO: remove and use toTpetra()
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal,Node> > v = tSource.getTpetra_MultiVector();
#else
    RCP< const Tpetra::MultiVector< Scalar,Node> > v = tSource.getTpetra_MultiVector();
#endif
    this->getTpetra_MultiVector()->doImport(*v, toTpetra(exporter), toTpetra(CM));
    
  }
  
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
  doExport(const DistObject< Scalar, LocalOrdinal, GlobalOrdinal, Node > &dest, const Export<LocalOrdinal,GlobalOrdinal,Node>& exporter, CombineMode CM) {
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::    
  doExport(const DistObject< Scalar, Node > &dest, const Export<Node>& exporter, CombineMode CM) {
#endif
    XPETRA_MONITOR("TpetraMultiVector::doExport");
    
    XPETRA_DYNAMIC_CAST(const TpetraMultiVectorClass, dest, tDest, "Xpetra::TpetraMultiVector::doImport only accept Xpetra::TpetraMultiVector as input arguments."); //TODO: remove and use toTpetra()
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal,Node> > v = tDest.getTpetra_MultiVector();
#else
    RCP< const Tpetra::MultiVector< Scalar,Node> > v = tDest.getTpetra_MultiVector();
#endif
    this->getTpetra_MultiVector()->doExport(*v, toTpetra(exporter), toTpetra(CM));

    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
  replaceMap(const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& map) {
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::    
  replaceMap(const RCP<const Map<Node> >& map) {
#endif
    XPETRA_MONITOR("TpetraMultiVector::replaceMap");
    this->getTpetra_MultiVector()->replaceMap(toTpetra(map));
  }
  
//! TpetraMultiVector constructor to wrap a Tpetra::MultiVector objecT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
  TpetraMultiVector(const Teuchos::RCP<Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node> > &vec) : vec_(vec) { } //TODO removed const
#else
  template<class Scalar, class Node>
  TpetraMultiVector<Scalar,Node>::    
  TpetraMultiVector(const Teuchos::RCP<Tpetra::MultiVector< Scalar, Node> > &vec) : vec_(vec) { } //TODO removed const
#endif

  //! Get the underlying Tpetra multivector
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP< Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node> > 
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
#else
  template<class Scalar, class Node>
  RCP< Tpetra::MultiVector< Scalar, Node> > 
  TpetraMultiVector<Scalar,Node>::    
#endif
  getTpetra_MultiVector() const { return vec_; }
  
  //! Set seed for Random function.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::    
#endif
  setSeed(unsigned int seed) { XPETRA_MONITOR("TpetraMultiVector::seedrandom"); Teuchos::ScalarTraits< Scalar >::seedrandom(seed); }
  

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
    /// \brief Return an unmanaged non-const view of the local data on a specific device.
    /// \tparam TargetDeviceType The Kokkos Device type whose data to return.
    ///
    /// \warning DO NOT USE THIS FUNCTION! There is no reason why you are working directly
    ///          with the Xpetra::TpetraMultiVector object. To write a code which is independent
    ///          from the underlying linear algebra package you should always use the abstract class,
    ///          i.e. Xpetra::MultiVector!
    ///
    /// \warning Be aware that the view on the multivector data is non-persisting, i.e.
    ///          only valid as long as the multivector does not run of scope!
#if 0
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
  template<class Scalar, class Node>
#endif
  template<class TargetDeviceType>
  typename Kokkos::Impl::if_c<
      Kokkos::Impl::is_same<
        typename dual_view_type::t_dev_um::execution_space::memory_space,
        typename TargetDeviceType::memory_space>::value,
        typename dual_view_type::t_dev_um,
        typename dual_view_type::t_host_um>::type
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
#else
  TpetraMultiVector<Scalar,Node>::    
#endif
  getLocalView () const {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    return this->MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node >::template getLocalView<TargetDeviceType>();
#else
    return this->MultiVector< Scalar, Node >::template getLocalView<TargetDeviceType>();
#endif
  }
#endif
  
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  typename TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::dual_view_type::t_host_um 
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
#else
  template<class Scalar, class Node>
  typename TpetraMultiVector<Scalar,Node>::dual_view_type::t_host_um 
  TpetraMultiVector<Scalar,Node>::    
#endif
  getHostLocalView () const {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    return subview(vec_->template getLocalView<typename TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::dual_view_type::host_mirror_space> (),
#else
    return subview(vec_->template getLocalView<typename TpetraMultiVector<Scalar,Node>::dual_view_type::host_mirror_space> (),
#endif
                   Kokkos::ALL(), Kokkos::ALL());
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  typename TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::dual_view_type::t_dev_um 
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
#else
  template<class Scalar, class Node>
  typename TpetraMultiVector<Scalar,Node>::dual_view_type::t_dev_um 
  TpetraMultiVector<Scalar,Node>::    
#endif
  getDeviceLocalView() const {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    return subview(vec_->template getLocalView<typename TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::dual_view_type::t_dev_um::execution_space> (),
#else
    return subview(vec_->template getLocalView<typename TpetraMultiVector<Scalar,Node>::dual_view_type::t_dev_um::execution_space> (),
#endif
                   Kokkos::ALL(), Kokkos::ALL());
  }
  
#endif

  /// \brief Implementation of the assignment operator (operator=);
  ///   does a deep copy.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::    
  assign (const MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& rhs)
#else
  template<class Scalar, class Node>
  void TpetraMultiVector<Scalar,Node>::    
  assign (const MultiVector<Scalar, Node>& rhs)
#endif
  {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> this_type;
#else
    typedef TpetraMultiVector<Scalar, Node> this_type;
#endif
    const this_type* rhsPtr = dynamic_cast<const this_type*> (&rhs);
    TEUCHOS_TEST_FOR_EXCEPTION(
                               rhsPtr == NULL, std::invalid_argument, "Xpetra::MultiVector::operator=:"
                               " The left-hand side (LHS) of the assignment has a different type than "
                               "the right-hand side (RHS).  The LHS has type Xpetra::TpetraMultiVector"
                               " (which means it wraps a Tpetra::MultiVector), but the RHS has some "
                               "other type.  This probably means that the RHS wraps an "
                               "Epetra_MultiVector.  Xpetra::MultiVector does not currently implement "
                               "assignment from an Epetra object to a Tpetra object, though this could"
                               " be added with sufficient interest.");
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> TMV;
#else
    typedef Tpetra::MultiVector<Scalar, Node> TMV;
#endif
    RCP<const TMV> rhsImpl = rhsPtr->getTpetra_MultiVector ();
    RCP<TMV> lhsImpl = this->getTpetra_MultiVector ();
    
    TEUCHOS_TEST_FOR_EXCEPTION(
                               rhsImpl.is_null (), std::logic_error, "Xpetra::MultiVector::operator= "
                               "(in Xpetra::TpetraMultiVector::assign): *this (the right-hand side of "
                               "the assignment) has a null RCP<Tpetra::MultiVector> inside.  Please "
                               "report this bug to the Xpetra developers.");
    TEUCHOS_TEST_FOR_EXCEPTION(
                               lhsImpl.is_null (), std::logic_error, "Xpetra::MultiVector::operator= "
                               "(in Xpetra::TpetraMultiVector::assign): The left-hand side of the "
                               "assignment has a null RCP<Tpetra::MultiVector> inside.  Please report "
                               "this bug to the Xpetra developers.");
    
    Tpetra::deep_copy (*lhsImpl, *rhsImpl);
  }

  
#ifdef HAVE_XPETRA_EPETRA

#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))

  // specialization for TpetraMultiVector on EpetraNode and GO=int
  template <class Scalar>
  class TpetraMultiVector<Scalar,int,int,EpetraNode>
    : public virtual MultiVector< Scalar, int, int, EpetraNode >
  {
    typedef int LocalOrdinal;
    typedef int GlobalOrdinal;
    typedef EpetraNode Node;

    // The following typedef are used by the XPETRA_DYNAMIC_CAST() macro.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> TpetraMultiVectorClass;
#else
    typedef TpetraMultiVector<Scalar,Node> TpetraMultiVectorClass;
#endif

  public:

    //! @name Constructors and destructor
    //@{

    //! Default constructor
    TpetraMultiVector () {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Basic constuctor.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraMultiVector(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &map, size_t NumVectors, bool zeroOut=true) {
#else
    TpetraMultiVector(const Teuchos::RCP< const Map<Node > > &map, size_t NumVectors, bool zeroOut=true) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Copy constructor (performs a deep copy).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraMultiVector(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &source) {
#else
    TpetraMultiVector(const MultiVector< Scalar, Node > &source) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Create multivector by copying two-dimensional array of local data.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraMultiVector(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &map, const Teuchos::ArrayView< const Scalar > &A, size_t LDA, size_t NumVectors) {
#else
    TpetraMultiVector(const Teuchos::RCP< const Map<Node > > &map, const Teuchos::ArrayView< const Scalar > &A, size_t LDA, size_t NumVectors) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Create multivector by copying array of views of local data.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraMultiVector(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &map, const Teuchos::ArrayView< const Teuchos::ArrayView< const Scalar > > &ArrayOfPtrs, size_t NumVectors) {
#else
    TpetraMultiVector(const Teuchos::RCP< const Map<Node > > &map, const Teuchos::ArrayView< const Teuchos::ArrayView< const Scalar > > &ArrayOfPtrs, size_t NumVectors) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }


    //! Destructor (virtual for memory safety of derived classes).
    virtual ~TpetraMultiVector() {  }

    //@}

    //! @name Post-construction modification routines
    //@{

    //! Replace value, using global (row) index.
    void replaceGlobalValue(GlobalOrdinal globalRow, size_t vectorIndex, const Scalar &value) { }

    //! Add value to existing value, using global (row) index.
    void sumIntoGlobalValue(GlobalOrdinal globalRow, size_t vectorIndex, const Scalar &value) { }

    //! Replace value, using local (row) index.
    void replaceLocalValue(LocalOrdinal myRow, size_t vectorIndex, const Scalar &value) { }

    //! Add value to existing value, using local (row) index.
    void sumIntoLocalValue(LocalOrdinal myRow, size_t vectorIndex, const Scalar &value) { }

    //! Set all values in the multivector with the given value.
    void putScalar(const Scalar &value) { }

    //! Sum values of a locally replicated multivector across all processes.
    void reduce() { }

    //@}

    //! @name Data Copy and View get methods
    //@{

    //! Return a Vector which is a const view of column j.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP< const Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > > getVector(size_t j) const { return Teuchos::null; }
#else
    Teuchos::RCP< const Vector< Scalar, Node > > getVector(size_t j) const { return Teuchos::null; }
#endif

    //! Return a Vector which is a nonconst view of column j.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP< Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > > getVectorNonConst(size_t j) { return Teuchos::null; }
#else
    Teuchos::RCP< Vector< Scalar, Node > > getVectorNonConst(size_t j) { return Teuchos::null; }
#endif

    //! Const view of the local values in a particular vector of this multivector.
    Teuchos::ArrayRCP< const Scalar > getData(size_t j) const { return Teuchos::ArrayRCP< const Scalar >(); }

    //! View of the local values in a particular vector of this multivector.
    Teuchos::ArrayRCP< Scalar > getDataNonConst(size_t j) { return Teuchos::ArrayRCP< Scalar >(); }

    //! Fill the given array with a copy of this multivector's local values.
    void get1dCopy(Teuchos::ArrayView< Scalar > A, size_t LDA) const { }

    //! Fill the given array with a copy of this multivector's local values.
    void get2dCopy(Teuchos::ArrayView< const Teuchos::ArrayView< Scalar > > ArrayOfPtrs) const { }

    //! Const persisting (1-D) view of this multivector's local values.
    Teuchos::ArrayRCP< const Scalar > get1dView() const { return Teuchos::ArrayRCP< const Scalar >(); }

    //! Return const persisting pointers to values.
    Teuchos::ArrayRCP< Teuchos::ArrayRCP< const Scalar > > get2dView() const { return Teuchos::ArrayRCP< Teuchos::ArrayRCP< const Scalar > >(); }

    //! Nonconst persisting (1-D) view of this multivector's local values.
    Teuchos::ArrayRCP< Scalar > get1dViewNonConst() { return Teuchos::ArrayRCP< Scalar >(); }

    //! Return non-const persisting pointers to values.
    Teuchos::ArrayRCP< Teuchos::ArrayRCP< Scalar > > get2dViewNonConst() { return Teuchos::ArrayRCP< Teuchos::ArrayRCP< Scalar > >(); }

    //@}

    //! @name Mathematical methods
    //@{

    //! Compute dot product of each corresponding pair of vectors, dots[i] = this[i].dot(A[i]).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void dot(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const Teuchos::ArrayView< Scalar > &dots) const { }
#else
    void dot(const MultiVector< Scalar, Node > &A, const Teuchos::ArrayView< Scalar > &dots) const { }
#endif

    //! Put element-wise absolute values of input Multi-vector in target: A = abs(this).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void abs(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A) { }
#else
    void abs(const MultiVector< Scalar, Node > &A) { }
#endif

    //! Put element-wise reciprocal values of input Multi-vector in target, this(i,j) = 1/A(i,j).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void reciprocal(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A) { }
#else
    void reciprocal(const MultiVector< Scalar, Node > &A) { }
#endif

    //! Scale the current values of a multi-vector, this = alpha*this.
    void scale(const Scalar &alpha) { }

    //! Scale the current values of a multi-vector, this[j] = alpha[j]*this[j].
    void scale(Teuchos::ArrayView< const Scalar > alpha) { }

    //! Replace multi-vector values with scaled values of A, this = alpha*A.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void scale(const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A) { }
#else
    void scale(const Scalar &alpha, const MultiVector< Scalar, Node > &A) { }
#endif

    //! Update multi-vector values with scaled values of A, this = beta*this + alpha*A.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void update(const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const Scalar &beta) { }
#else
    void update(const Scalar &alpha, const MultiVector< Scalar, Node > &A, const Scalar &beta) { }
#endif

    //! Update multi-vector with scaled values of A and B, this = gamma*this + alpha*A + beta*B.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void update(const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const Scalar &beta, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &B, const Scalar &gamma) { }
#else
    void update(const Scalar &alpha, const MultiVector< Scalar, Node > &A, const Scalar &beta, const MultiVector< Scalar, Node > &B, const Scalar &gamma) { }
#endif

    //! Compute 1-norm of each vector in multi-vector.
    void norm1(const Teuchos::ArrayView< typename Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const { }

    //!
    void norm2(const Teuchos::ArrayView< typename Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const { }

    //! Compute Inf-norm of each vector in multi-vector.
    void normInf(const Teuchos::ArrayView< typename Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const { }

    //! Compute mean (average) value of each vector in multi-vector. The outcome of this routine is undefined for non-floating point scalar types (e.g., int).
    void meanValue(const Teuchos::ArrayView< Scalar > &means) const { }

    //! Matrix-matrix multiplication: this = beta*this + alpha*op(A)*op(B).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void multiply(Teuchos::ETransp transA, Teuchos::ETransp transB, const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &B, const Scalar &beta) { }
#else
    void multiply(Teuchos::ETransp transA, Teuchos::ETransp transB, const Scalar &alpha, const MultiVector< Scalar, Node > &A, const MultiVector< Scalar, Node > &B, const Scalar &beta) { }
#endif

    //@}

    //! @name Attribute access functions
    //@{

    //! Number of columns in the multivector.
    size_t getNumVectors() const { return 0; }

    //! Local number of rows on the calling process.
    size_t getLocalLength() const { return 0; }

    //! Global number of rows in the multivector.
    global_size_t getGlobalLength() const { return 0; }

    // \! Checks to see if the local length, number of vectors and size of Scalar type match
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    bool isSameSize(const MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> & vec) const { return false; }
#else
    bool isSameSize(const MultiVector<Scalar,Node> & vec) const { return false; }
#endif
  
    //@}

    //! @name Overridden from Teuchos::Describable
    //@{

    //! A simple one-line description of this object.
    std::string description() const { return std::string(""); }

    //! Print the object with the given verbosity level to a FancyOStream.
    void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel=Teuchos::Describable::verbLevel_default) const { }

    //@}

    //! Element-wise multiply of a Vector A with a TpetraMultiVector B.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void elementWiseMultiply(Scalar scalarAB, const Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &A, const MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &B, Scalar scalarThis) {};
#else
    void elementWiseMultiply(Scalar scalarAB, const Vector<Scalar,Node> &A, const MultiVector<Scalar,Node> &B, Scalar scalarThis) {};
#endif

    //! Set multi-vector values to random numbers.
    void randomize(bool bUseXpetraImplementation = false) { }

    //{@
    // Implements DistObject interface

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP< const Map<LocalOrdinal,GlobalOrdinal,Node> > getMap() const { return Teuchos::null; }
#else
    Teuchos::RCP< const Map<Node> > getMap() const { return Teuchos::null; }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doImport(const DistObject< Scalar, LocalOrdinal,GlobalOrdinal,Node> &source, const Import<LocalOrdinal,GlobalOrdinal,Node> &importer, CombineMode CM) { }
#else
    void doImport(const DistObject< Scalar,Node> &source, const Import<Node> &importer, CombineMode CM) { }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doExport(const DistObject< Scalar, LocalOrdinal, GlobalOrdinal, Node > &dest, const Import<LocalOrdinal,GlobalOrdinal,Node>& importer, CombineMode CM) { }
#else
    void doExport(const DistObject< Scalar, Node > &dest, const Import<Node>& importer, CombineMode CM) { }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doImport(const DistObject< Scalar, LocalOrdinal, GlobalOrdinal, Node > &source, const Export<LocalOrdinal,GlobalOrdinal,Node>& exporter, CombineMode CM) { }
#else
    void doImport(const DistObject< Scalar, Node > &source, const Export<Node>& exporter, CombineMode CM) { }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doExport(const DistObject< Scalar, LocalOrdinal, GlobalOrdinal, Node > &dest, const Export<LocalOrdinal,GlobalOrdinal,Node>& exporter, CombineMode CM) { }
#else
    void doExport(const DistObject< Scalar, Node > &dest, const Export<Node>& exporter, CombineMode CM) { }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void replaceMap(const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& map) { }
#else
    void replaceMap(const RCP<const Map<Node> >& map) { }
#endif

//@}

    //! @name Xpetra specific
    //@{

    //! TpetraMultiVector constructor to wrap a Tpetra::MultiVector object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraMultiVector(const Teuchos::RCP<Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node> > &vec) {
#else
    TpetraMultiVector(const Teuchos::RCP<Tpetra::MultiVector< Scalar, Node> > &vec) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Get the underlying Tpetra multivector
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node> > getTpetra_MultiVector() const { return Teuchos::null; }
#else
    RCP< Tpetra::MultiVector< Scalar, Node> > getTpetra_MultiVector() const { return Teuchos::null; }
#endif

    //! Set seed for Random function.
    void setSeed(unsigned int seed) { }


#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef typename Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::dual_view_type dual_view_type;
#else
    typedef typename Xpetra::MultiVector<Scalar, Node>::dual_view_type dual_view_type;
#endif
    //typedef typename Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::host_execution_space host_execution_space;
    //typedef typename Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::dev_execution_space dev_execution_space;

    /// \brief Return an unmanaged non-const view of the local data on a specific device.
    /// \tparam TargetDeviceType The Kokkos Device type whose data to return.
    ///
    /// \warning DO NOT USE THIS FUNCTION! There is no reason why you are working directly
    ///          with the Xpetra::TpetraMultiVector object. To write a code which is independent
    ///          from the underlying linear algebra package you should always use the abstract class,
    ///          i.e. Xpetra::MultiVector!
    ///
    /// \warning Be aware that the view on the multivector data is non-persisting, i.e.
    ///          only valid as long as the multivector does not run of scope!
    template<class TargetDeviceType>
    typename Kokkos::Impl::if_c<
      Kokkos::Impl::is_same<
        typename dual_view_type::t_dev_um::execution_space::memory_space,
        typename TargetDeviceType::memory_space>::value,
        typename dual_view_type::t_dev_um,
        typename dual_view_type::t_host_um>::type
    getLocalView () const {
      typename Kokkos::Impl::if_c<
            Kokkos::Impl::is_same<
              typename dual_view_type::t_dev_um::execution_space::memory_space,
              typename TargetDeviceType::memory_space>::value,
              typename dual_view_type::t_dev_um,
              typename dual_view_type::t_host_um>::type dummy;
      return dummy;
    }

    typename dual_view_type::t_host_um getHostLocalView () const {
      //return subview(vec_->template getLocalView<typename dual_view_type::host_mirror_space> (),
      //    Kokkos::ALL(), Kokkos::ALL());
      return typename dual_view_type::t_host_um();
    }

    typename dual_view_type::t_dev_um getDeviceLocalView() const {
      //return subview(vec_->template getLocalView<typename dual_view_type::t_dev_um::execution_space> (),
      //    Kokkos::ALL(), Kokkos::ALL());
      return typename dual_view_type::t_dev_um();
    }

#endif

    //@}

  protected:
    /// \brief Implementation of the assignment operator (operator=);
    ///   does a deep copy.
    virtual void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    assign (const MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& rhs)
#else
    assign (const MultiVector<Scalar, Node>& rhs)
#endif
    { }
  }; // TpetraMultiVector class (specialization GO=int, NO=EpetraNode)
#endif

#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_LONG_LONG))))

  // specialization for TpetraMultiVector on EpetraNode and GO=long long
  template <class Scalar>
  class TpetraMultiVector<Scalar,int,long long,EpetraNode>
    : public virtual MultiVector< Scalar, int, long long, EpetraNode >
  {
    typedef int LocalOrdinal;
    typedef long long GlobalOrdinal;
    typedef EpetraNode Node;

    // The following typedef are used by the XPETRA_DYNAMIC_CAST() macro.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> TpetraMultiVectorClass;
#else
    typedef TpetraMultiVector<Scalar,Node> TpetraMultiVectorClass;
#endif

  public:

    //! @name Constructors and destructor
    //@{

    //! Basic constuctor.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraMultiVector(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &map, size_t NumVectors, bool zeroOut=true) {
#else
    TpetraMultiVector(const Teuchos::RCP< const Map<Node > > &map, size_t NumVectors, bool zeroOut=true) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

    //! Copy constructor (performs a deep copy).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraMultiVector(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &source) {
#else
    TpetraMultiVector(const MultiVector< Scalar, Node > &source) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

    //! Create multivector by copying two-dimensional array of local data.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraMultiVector(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &map, const Teuchos::ArrayView< const Scalar > &A, size_t LDA, size_t NumVectors) {
#else
    TpetraMultiVector(const Teuchos::RCP< const Map<Node > > &map, const Teuchos::ArrayView< const Scalar > &A, size_t LDA, size_t NumVectors) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

    //! Create multivector by copying array of views of local data.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraMultiVector(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &map, const Teuchos::ArrayView< const Teuchos::ArrayView< const Scalar > > &ArrayOfPtrs, size_t NumVectors) {
#else
    TpetraMultiVector(const Teuchos::RCP< const Map<Node > > &map, const Teuchos::ArrayView< const Teuchos::ArrayView< const Scalar > > &ArrayOfPtrs, size_t NumVectors) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }


    //! Destructor (virtual for memory safety of derived classes).
    virtual ~TpetraMultiVector() {  }

    //@}

    //! @name Post-construction modification routines
    //@{

    //! Replace value, using global (row) index.
    void replaceGlobalValue(GlobalOrdinal globalRow, size_t vectorIndex, const Scalar &value) { }

    //! Add value to existing value, using global (row) index.
    void sumIntoGlobalValue(GlobalOrdinal globalRow, size_t vectorIndex, const Scalar &value) { }

    //! Replace value, using local (row) index.
    void replaceLocalValue(LocalOrdinal myRow, size_t vectorIndex, const Scalar &value) { }

    //! Add value to existing value, using local (row) index.
    void sumIntoLocalValue(LocalOrdinal myRow, size_t vectorIndex, const Scalar &value) { }

    //! Set all values in the multivector with the given value.
    void putScalar(const Scalar &value) { }

    //! Sum values of a locally replicated multivector across all processes.
    void reduce() { }

    //@}

    //! @name Data Copy and View get methods
    //@{

    //! Return a Vector which is a const view of column j.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP< const Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > > getVector(size_t j) const { return Teuchos::null; }
#else
    Teuchos::RCP< const Vector< Scalar, Node > > getVector(size_t j) const { return Teuchos::null; }
#endif

    //! Return a Vector which is a nonconst view of column j.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP< Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > > getVectorNonConst(size_t j) { return Teuchos::null; }
#else
    Teuchos::RCP< Vector< Scalar, Node > > getVectorNonConst(size_t j) { return Teuchos::null; }
#endif

    //! Const view of the local values in a particular vector of this multivector.
    Teuchos::ArrayRCP< const Scalar > getData(size_t j) const { return Teuchos::ArrayRCP< const Scalar >(); }

    //! View of the local values in a particular vector of this multivector.
    Teuchos::ArrayRCP< Scalar > getDataNonConst(size_t j) { return Teuchos::ArrayRCP< Scalar >(); }

    //! Fill the given array with a copy of this multivector's local values.
    void get1dCopy(Teuchos::ArrayView< Scalar > A, size_t LDA) const { }

    //! Fill the given array with a copy of this multivector's local values.
    void get2dCopy(Teuchos::ArrayView< const Teuchos::ArrayView< Scalar > > ArrayOfPtrs) const { }

    //! Const persisting (1-D) view of this multivector's local values.
    Teuchos::ArrayRCP< const Scalar > get1dView() const { return Teuchos::ArrayRCP< const Scalar >(); }

    //! Return const persisting pointers to values.
    Teuchos::ArrayRCP< Teuchos::ArrayRCP< const Scalar > > get2dView() const { return Teuchos::ArrayRCP< Teuchos::ArrayRCP< const Scalar > >(); }

    //! Nonconst persisting (1-D) view of this multivector's local values.
    Teuchos::ArrayRCP< Scalar > get1dViewNonConst() { return Teuchos::ArrayRCP< Scalar >(); }

    //! Return non-const persisting pointers to values.
    Teuchos::ArrayRCP< Teuchos::ArrayRCP< Scalar > > get2dViewNonConst() { return Teuchos::ArrayRCP< Teuchos::ArrayRCP< Scalar > >(); }

    //@}

    //! @name Mathematical methods
    //@{

    //! Compute dot product of each corresponding pair of vectors, dots[i] = this[i].dot(A[i]).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void dot(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const Teuchos::ArrayView< Scalar > &dots) const { }
#else
    void dot(const MultiVector< Scalar, Node > &A, const Teuchos::ArrayView< Scalar > &dots) const { }
#endif

    //! Put element-wise absolute values of input Multi-vector in target: A = abs(this).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void abs(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A) { }
#else
    void abs(const MultiVector< Scalar, Node > &A) { }
#endif

    //! Put element-wise reciprocal values of input Multi-vector in target, this(i,j) = 1/A(i,j).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void reciprocal(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A) { }
#else
    void reciprocal(const MultiVector< Scalar, Node > &A) { }
#endif

    //! Scale the current values of a multi-vector, this = alpha*this.
    void scale(const Scalar &alpha) { }

    //! Scale the current values of a multi-vector, this[j] = alpha[j]*this[j].
    void scale(Teuchos::ArrayView< const Scalar > alpha) { }

    //! Replace multi-vector values with scaled values of A, this = alpha*A.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void scale(const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A) { }
#else
    void scale(const Scalar &alpha, const MultiVector< Scalar, Node > &A) { }
#endif

    //! Update multi-vector values with scaled values of A, this = beta*this + alpha*A.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void update(const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const Scalar &beta) { }
#else
    void update(const Scalar &alpha, const MultiVector< Scalar, Node > &A, const Scalar &beta) { }
#endif

    //! Update multi-vector with scaled values of A and B, this = gamma*this + alpha*A + beta*B.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void update(const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const Scalar &beta, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &B, const Scalar &gamma) { }
#else
    void update(const Scalar &alpha, const MultiVector< Scalar, Node > &A, const Scalar &beta, const MultiVector< Scalar, Node > &B, const Scalar &gamma) { }
#endif

    //! Compute 1-norm of each vector in multi-vector.
    void norm1(const Teuchos::ArrayView< typename Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const { }

    //!
    void norm2(const Teuchos::ArrayView< typename Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const { }

    //! Compute Inf-norm of each vector in multi-vector.
    void normInf(const Teuchos::ArrayView< typename Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const { }

    //! Compute mean (average) value of each vector in multi-vector. The outcome of this routine is undefined for non-floating point scalar types (e.g., int).
    void meanValue(const Teuchos::ArrayView< Scalar > &means) const { }

    //! Matrix-matrix multiplication: this = beta*this + alpha*op(A)*op(B).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void multiply(Teuchos::ETransp transA, Teuchos::ETransp transB, const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &B, const Scalar &beta) { }
#else
    void multiply(Teuchos::ETransp transA, Teuchos::ETransp transB, const Scalar &alpha, const MultiVector< Scalar, Node > &A, const MultiVector< Scalar, Node > &B, const Scalar &beta) { }
#endif

    //@}

    //! @name Attribute access functions
    //@{

    //! Number of columns in the multivector.
    size_t getNumVectors() const { return 0; }

    //! Local number of rows on the calling process.
    size_t getLocalLength() const { return 0; }

    //! Global number of rows in the multivector.
    global_size_t getGlobalLength() const { return 0; }

    // \! Checks to see if the local length, number of vectors and size of Scalar type match
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    bool isSameSize(const MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> & vec) const { return false; }
#else
    bool isSameSize(const MultiVector<Scalar,Node> & vec) const { return false; }
#endif

    //@}

    //! @name Overridden from Teuchos::Describable
    //@{

    //! A simple one-line description of this object.
    std::string description() const { return std::string(""); }

    //! Print the object with the given verbosity level to a FancyOStream.
    void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel=Teuchos::Describable::verbLevel_default) const { }

    //@}

    //! Element-wise multiply of a Vector A with a TpetraMultiVector B.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void elementWiseMultiply(Scalar scalarAB, const Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &A, const MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &B, Scalar scalarThis) {};
#else
    void elementWiseMultiply(Scalar scalarAB, const Vector<Scalar,Node> &A, const MultiVector<Scalar,Node> &B, Scalar scalarThis) {};
#endif

    //! Set multi-vector values to random numbers.
    void randomize(bool bUseXpetraImplementation = false) { }

    //{@
    // Implements DistObject interface

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP< const Map<LocalOrdinal,GlobalOrdinal,Node> > getMap() const { return Teuchos::null; }
#else
    Teuchos::RCP< const Map<Node> > getMap() const { return Teuchos::null; }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doImport(const DistObject< Scalar, LocalOrdinal,GlobalOrdinal,Node> &source, const Import<LocalOrdinal,GlobalOrdinal,Node> &importer, CombineMode CM) { }
#else
    void doImport(const DistObject< Scalar,Node> &source, const Import<Node> &importer, CombineMode CM) { }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doExport(const DistObject< Scalar, LocalOrdinal, GlobalOrdinal, Node > &dest, const Import<LocalOrdinal,GlobalOrdinal,Node>& importer, CombineMode CM) { }
#else
    void doExport(const DistObject< Scalar, Node > &dest, const Import<Node>& importer, CombineMode CM) { }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doImport(const DistObject< Scalar, LocalOrdinal, GlobalOrdinal, Node > &source, const Export<LocalOrdinal,GlobalOrdinal,Node>& exporter, CombineMode CM) { }
#else
    void doImport(const DistObject< Scalar, Node > &source, const Export<Node>& exporter, CombineMode CM) { }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doExport(const DistObject< Scalar, LocalOrdinal, GlobalOrdinal, Node > &dest, const Export<LocalOrdinal,GlobalOrdinal,Node>& exporter, CombineMode CM) { }
#else
    void doExport(const DistObject< Scalar, Node > &dest, const Export<Node>& exporter, CombineMode CM) { }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void replaceMap(const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& map) { }
#else
    void replaceMap(const RCP<const Map<Node> >& map) { }
#endif

//@}

    //! @name Xpetra specific
    //@{

    //! TpetraMultiVector constructor to wrap a Tpetra::MultiVector object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraMultiVector(const Teuchos::RCP<Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node> > &vec) {
#else
    TpetraMultiVector(const Teuchos::RCP<Tpetra::MultiVector< Scalar, Node> > &vec) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

    //! Get the underlying Tpetra multivector
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node> > getTpetra_MultiVector() const { return Teuchos::null; }
#else
    RCP< Tpetra::MultiVector< Scalar, Node> > getTpetra_MultiVector() const { return Teuchos::null; }
#endif

    //! Set seed for Random function.
    void setSeed(unsigned int seed) { }


#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef typename Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::dual_view_type dual_view_type;
#else
    typedef typename Xpetra::MultiVector<Scalar, Node>::dual_view_type dual_view_type;
#endif
    //typedef typename Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::host_execution_space host_execution_space;
    //typedef typename Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::dev_execution_space dev_execution_space;

    /// \brief Return an unmanaged non-const view of the local data on a specific device.
    /// \tparam TargetDeviceType The Kokkos Device type whose data to return.
    ///
    /// \warning DO NOT USE THIS FUNCTION! There is no reason why you are working directly
    ///          with the Xpetra::TpetraMultiVector object. To write a code which is independent
    ///          from the underlying linear algebra package you should always use the abstract class,
    ///          i.e. Xpetra::MultiVector!
    ///
    /// \warning Be aware that the view on the multivector data is non-persisting, i.e.
    ///          only valid as long as the multivector does not run of scope!
    template<class TargetDeviceType>
    typename Kokkos::Impl::if_c<
      Kokkos::Impl::is_same<
        typename dual_view_type::t_dev_um::execution_space::memory_space,
        typename TargetDeviceType::memory_space>::value,
        typename dual_view_type::t_dev_um,
        typename dual_view_type::t_host_um>::type
    getLocalView () const {
      typename Kokkos::Impl::if_c<
            Kokkos::Impl::is_same<
              typename dual_view_type::t_dev_um::execution_space::memory_space,
              typename TargetDeviceType::memory_space>::value,
              typename dual_view_type::t_dev_um,
              typename dual_view_type::t_host_um>::type dummy;
      return dummy;
    }

    typename dual_view_type::t_host_um getHostLocalView () const {
      //return subview(vec_->template getLocalView<typename dual_view_type::host_mirror_space> (),
      //    Kokkos::ALL(), Kokkos::ALL());
      return typename dual_view_type::t_host_um();
    }

    typename dual_view_type::t_dev_um getDeviceLocalView() const {
      //return subview(vec_->template getLocalView<typename dual_view_type::t_dev_um::execution_space> (),
      //    Kokkos::ALL(), Kokkos::ALL());
      return typename dual_view_type::t_dev_um();
    }

#endif

    //@}

  protected:
    /// \brief Implementation of the assignment operator (operator=);
    ///   does a deep copy.
    virtual void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    assign (const MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& rhs)
#else
    assign (const MultiVector<Scalar, Node>& rhs)
#endif
    { }
  }; // TpetraMultiVector class (specialization GO=int, NO=EpetraNode)

#endif // TpetraMultiVector class (specialization GO=long long, NO=EpetraNode)

#endif // HAVE_XPETRA_EPETRA

} // Xpetra namespace

// Following header file inculsion is needed for the dynamic_cast to TpetraVector in 
// elementWiseMultiply (because we cannot dynamic_cast if target is not a complete type)
// It is included here to avoid circular dependency between Vector and MultiVector
// TODO: there is certainly a more elegant solution...
#include "Xpetra_TpetraVector.hpp"

namespace Xpetra {

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
  template <class Scalar, class Node>
#endif
  void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  TpetraMultiVector<Scalar,Node>::
#endif
  elementWiseMultiply(Scalar scalarAB, 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                      const Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &A, 
                      const MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &B, 
#else
                      const Vector<Scalar,Node> &A, 
                      const MultiVector<Scalar,Node> &B, 
#endif
                      Scalar scalarThis) 
  {
    XPETRA_MONITOR("TpetraMultiVector::elementWiseMultiply");

    // XPETRA_DYNAMIC_CAST won't take TpetraVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>
    // as an argument, hence the following typedef.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef TpetraVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> tpv;
#else
    typedef TpetraVector<Scalar,Node> tpv;
#endif
    XPETRA_DYNAMIC_CAST(const tpv, A, tA, "Xpetra::TpetraMultiVectorMatrix->multiply() only accept Xpetra::TpetraMultiVector as input arguments.");
    XPETRA_DYNAMIC_CAST(const TpetraMultiVector, B, tB, "Xpetra::TpetraMultiVectorMatrix->multiply() only accept Xpetra::TpetraMultiVector as input arguments.");
    vec_->elementWiseMultiply(scalarAB, *tA.getTpetra_Vector(), *tB.getTpetra_MultiVector(), scalarThis);
  }

} // Xpetra namespace

#define XPETRA_TPETRAMULTIVECTOR_SHORT
#endif // XPETRA_TPETRAMULTIVECTOR_HPP
