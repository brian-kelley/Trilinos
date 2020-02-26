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

// WARNING: This code is experimental. Backwards compatibility should not be expected.

#ifndef XPETRA_CRSMATRIXWRAP_DEF_HPP
#define XPETRA_CRSMATRIXWRAP_DEF_HPP

#include <Kokkos_DefaultNode.hpp>

#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_Hashtable.hpp>

#include "Xpetra_ConfigDefs.hpp"
#include "Xpetra_Exceptions.hpp"

#include "Xpetra_MultiVector.hpp"
#include "Xpetra_CrsGraph.hpp"
#include "Xpetra_CrsMatrix.hpp"
#include "Xpetra_CrsMatrixFactory.hpp"

#include "Xpetra_Matrix.hpp"

#include "Xpetra_CrsMatrixWrap_decl.hpp"

namespace Xpetra {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::CrsMatrixWrap (const RCP<const Map>& rowMap)
#else
  template <class Scalar, class Node>
  CrsMatrixWrap<Scalar,Node>::CrsMatrixWrap (const RCP<const Map>& rowMap)
#endif
    : finalDefaultView_ (false)
  {
    matrixData_ = CrsMatrixFactory::Build (rowMap);
    CreateDefaultView ();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::CrsMatrixWrap (const RCP<const Map>& rowMap,
#else
  template <class Scalar, class Node>
  CrsMatrixWrap<Scalar,Node>::CrsMatrixWrap (const RCP<const Map>& rowMap,
#endif
                 size_t maxNumEntriesPerRow)
    : finalDefaultView_ (false)
  {
    matrixData_ = CrsMatrixFactory::Build (rowMap, maxNumEntriesPerRow);
    CreateDefaultView ();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::CrsMatrixWrap (const RCP<const Map>& rowMap,
#else
  template <class Scalar, class Node>
  CrsMatrixWrap<Scalar,Node>::CrsMatrixWrap (const RCP<const Map>& rowMap,
#endif
                 const ArrayRCP<const size_t>& NumEntriesPerRowToAlloc)
    : finalDefaultView_ (false)
  {
    matrixData_ = CrsMatrixFactory::Build(rowMap, NumEntriesPerRowToAlloc);
    CreateDefaultView ();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::CrsMatrixWrap(const RCP<const Map> &rowMap, const RCP<const Map>& colMap, size_t maxNumEntriesPerRow)
#else
  template <class Scalar, class Node>
  CrsMatrixWrap<Scalar,Node>::CrsMatrixWrap(const RCP<const Map> &rowMap, const RCP<const Map>& colMap, size_t maxNumEntriesPerRow)
#endif
    : finalDefaultView_(false)
  {
    // Set matrix data
    matrixData_ = CrsMatrixFactory::Build(rowMap, colMap, maxNumEntriesPerRow);

    // Default view
    CreateDefaultView();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::CrsMatrixWrap(const RCP<const Map> &rowMap, const RCP<const Map>& colMap, const ArrayRCP<const size_t> &NumEntriesPerRowToAlloc)
#else
  template <class Scalar, class Node>
  CrsMatrixWrap<Scalar,Node>::CrsMatrixWrap(const RCP<const Map> &rowMap, const RCP<const Map>& colMap, const ArrayRCP<const size_t> &NumEntriesPerRowToAlloc)
#endif
    : finalDefaultView_(false)
  {
    // Set matrix data
    matrixData_ = CrsMatrixFactory::Build(rowMap, colMap, NumEntriesPerRowToAlloc);

    // Default view
    CreateDefaultView();
  }

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef HAVE_XPETRA_TPETRA
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::CrsMatrixWrap(const RCP<const Map> &rowMap, const RCP<const Map>& colMap, const local_matrix_type& lclMatrix, const Teuchos::RCP<Teuchos::ParameterList>& params)
#else
  template <class Scalar, class Node>
  CrsMatrixWrap<Scalar,Node>::CrsMatrixWrap(const RCP<const Map> &rowMap, const RCP<const Map>& colMap, const local_matrix_type& lclMatrix, const Teuchos::RCP<Teuchos::ParameterList>& params)
#endif
    : finalDefaultView_(false)
  {
    // Set matrix data
    matrixData_ = CrsMatrixFactory::Build(rowMap, colMap, lclMatrix, params);

    // Default view
    CreateDefaultView();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::CrsMatrixWrap(const local_matrix_type& lclMatrix, const RCP<const Map> &rowMap, const RCP<const Map>& colMap,
#else
  template <class Scalar, class Node>
  CrsMatrixWrap<Scalar,Node>::CrsMatrixWrap(const local_matrix_type& lclMatrix, const RCP<const Map> &rowMap, const RCP<const Map>& colMap,
#endif
                const RCP<const Map>& domainMap, const RCP<const Map>& rangeMap,
                const Teuchos::RCP<Teuchos::ParameterList>& params)
    : finalDefaultView_(false)
  {
    // Set matrix data
    matrixData_ = CrsMatrixFactory::Build(lclMatrix, rowMap, colMap, domainMap, rangeMap, params);

    // Default view
    CreateDefaultView();
  }
#else
#ifdef __GNUC__
#warning "Xpetra Kokkos interface for CrsMatrix is enabled (HAVE_XPETRA_KOKKOS_REFACTOR) but Tpetra is disabled. The Kokkos interface needs Tpetra to be enabled, too."
#endif
#endif
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::CrsMatrixWrap(RCP<CrsMatrix> matrix)
#else
  template <class Scalar, class Node>
  CrsMatrixWrap<Scalar,Node>::CrsMatrixWrap(RCP<CrsMatrix> matrix)
#endif
    : finalDefaultView_(matrix->isFillComplete())
  {
    // Set matrix data
    matrixData_ = matrix;

    // Default view
    CreateDefaultView();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::CrsMatrixWrap(const RCP<const CrsGraph>& graph, const RCP<ParameterList>& paramList)
#else
  template <class Scalar, class Node>
  CrsMatrixWrap<Scalar,Node>::CrsMatrixWrap(const RCP<const CrsGraph>& graph, const RCP<ParameterList>& paramList)
#endif
    : finalDefaultView_(false)
  {
    // Set matrix data
    matrixData_ = CrsMatrixFactory::Build(graph, paramList);

    // Default view
    CreateDefaultView();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::~CrsMatrixWrap() {}
#else
  template <class Scalar, class Node>
  CrsMatrixWrap<Scalar,Node>::~CrsMatrixWrap() {}
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::insertGlobalValues(GlobalOrdinal globalRow, const ArrayView<const GlobalOrdinal> &cols, const ArrayView<const Scalar> &vals) {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::insertGlobalValues(GlobalOrdinal globalRow, const ArrayView<const GlobalOrdinal> &cols, const ArrayView<const Scalar> &vals) {
#endif
    matrixData_->insertGlobalValues(globalRow, cols, vals);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::insertLocalValues(LocalOrdinal localRow, const ArrayView<const LocalOrdinal> &cols, const ArrayView<const Scalar> &vals) {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::insertLocalValues(LocalOrdinal localRow, const ArrayView<const LocalOrdinal> &cols, const ArrayView<const Scalar> &vals) {
#endif
    matrixData_->insertLocalValues(localRow, cols, vals);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::replaceGlobalValues(GlobalOrdinal globalRow,
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::replaceGlobalValues(GlobalOrdinal globalRow,
#endif
                           const ArrayView<const GlobalOrdinal> &cols,
                           const ArrayView<const Scalar>        &vals) { matrixData_->replaceGlobalValues(globalRow, cols, vals); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::replaceLocalValues(LocalOrdinal localRow,
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::replaceLocalValues(LocalOrdinal localRow,
#endif
                          const ArrayView<const LocalOrdinal> &cols,
                          const ArrayView<const Scalar>       &vals) { matrixData_->replaceLocalValues(localRow, cols, vals); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::setAllToScalar(const Scalar &alpha) { matrixData_->setAllToScalar(alpha); }
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::setAllToScalar(const Scalar &alpha) { matrixData_->setAllToScalar(alpha); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::scale(const Scalar &alpha) {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::scale(const Scalar &alpha) {
#endif
    matrixData_->scale(alpha);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::resumeFill(const RCP< ParameterList > &params) {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::resumeFill(const RCP< ParameterList > &params) {
#endif
    matrixData_->resumeFill(params);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::fillComplete(const RCP<const Map> &domainMap, const RCP<const Map> &rangeMap, const RCP<Teuchos::ParameterList> &params) {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::fillComplete(const RCP<const Map> &domainMap, const RCP<const Map> &rangeMap, const RCP<Teuchos::ParameterList> &params) {
#endif
    matrixData_->fillComplete(domainMap, rangeMap, params);

    // Update default view with the colMap because colMap can be <tt>null</tt> until fillComplete() is called.
    updateDefaultView();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::fillComplete(const RCP<ParameterList> &params) {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::fillComplete(const RCP<ParameterList> &params) {
#endif
    matrixData_->fillComplete(params);

    // Update default view with the colMap because colMap can be <tt>null</tt> until fillComplete() is called.
    updateDefaultView();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  global_size_t CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getGlobalNumRows() const {
#else
  template <class Scalar, class Node>
  global_size_t CrsMatrixWrap<Scalar,Node>::getGlobalNumRows() const {
#endif
    return matrixData_->getGlobalNumRows();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  global_size_t CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getGlobalNumCols() const {
#else
  template <class Scalar, class Node>
  global_size_t CrsMatrixWrap<Scalar,Node>::getGlobalNumCols() const {
#endif
    return matrixData_->getGlobalNumCols();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  size_t CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getNodeNumRows() const {
#else
  template <class Scalar, class Node>
  size_t CrsMatrixWrap<Scalar,Node>::getNodeNumRows() const {
#endif
    return matrixData_->getNodeNumRows();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  global_size_t CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getGlobalNumEntries() const {
#else
  template <class Scalar, class Node>
  global_size_t CrsMatrixWrap<Scalar,Node>::getGlobalNumEntries() const {
#endif
    return matrixData_->getGlobalNumEntries();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  size_t CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getNodeNumEntries() const {
#else
  template <class Scalar, class Node>
  size_t CrsMatrixWrap<Scalar,Node>::getNodeNumEntries() const {
#endif
    return matrixData_->getNodeNumEntries();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  size_t CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getNumEntriesInLocalRow(LocalOrdinal localRow) const {
#else
  template <class Scalar, class Node>
  size_t CrsMatrixWrap<Scalar,Node>::getNumEntriesInLocalRow(LocalOrdinal localRow) const {
#endif
    return matrixData_->getNumEntriesInLocalRow(localRow);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  size_t CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getNumEntriesInGlobalRow(GlobalOrdinal globalRow) const {
#else
  template <class Scalar, class Node>
  size_t CrsMatrixWrap<Scalar,Node>::getNumEntriesInGlobalRow(GlobalOrdinal globalRow) const {
#endif
    return matrixData_->getNumEntriesInGlobalRow(globalRow);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  size_t CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getGlobalMaxNumRowEntries() const {
#else
  template <class Scalar, class Node>
  size_t CrsMatrixWrap<Scalar,Node>::getGlobalMaxNumRowEntries() const {
#endif
    return matrixData_->getGlobalMaxNumRowEntries();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  size_t CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getNodeMaxNumRowEntries() const {
#else
  template <class Scalar, class Node>
  size_t CrsMatrixWrap<Scalar,Node>::getNodeMaxNumRowEntries() const {
#endif
    return matrixData_->getNodeMaxNumRowEntries();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  bool CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::isLocallyIndexed() const {
#else
  template <class Scalar, class Node>
  bool CrsMatrixWrap<Scalar,Node>::isLocallyIndexed() const {
#endif
    return matrixData_->isLocallyIndexed();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  bool CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::isGloballyIndexed() const {
#else
  template <class Scalar, class Node>
  bool CrsMatrixWrap<Scalar,Node>::isGloballyIndexed() const {
#endif
    return matrixData_->isGloballyIndexed();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  bool CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::isFillComplete() const {
#else
  template <class Scalar, class Node>
  bool CrsMatrixWrap<Scalar,Node>::isFillComplete() const {
#endif
    return matrixData_->isFillComplete();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getLocalRowCopy(LocalOrdinal LocalRow,
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::getLocalRowCopy(LocalOrdinal LocalRow,
#endif
                       const ArrayView<LocalOrdinal> &Indices,
                       const ArrayView<Scalar> &Values,
                       size_t &NumEntries
                       ) const {
    matrixData_->getLocalRowCopy(LocalRow, Indices, Values, NumEntries);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getGlobalRowView(GlobalOrdinal GlobalRow, ArrayView<const GlobalOrdinal> &indices, ArrayView<const Scalar> &values) const {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::getGlobalRowView(GlobalOrdinal GlobalRow, ArrayView<const GlobalOrdinal> &indices, ArrayView<const Scalar> &values) const {
#endif
     matrixData_->getGlobalRowView(GlobalRow, indices, values);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getLocalRowView(LocalOrdinal LocalRow, ArrayView<const LocalOrdinal> &indices, ArrayView<const Scalar> &values) const {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::getLocalRowView(LocalOrdinal LocalRow, ArrayView<const LocalOrdinal> &indices, ArrayView<const Scalar> &values) const {
#endif
     matrixData_->getLocalRowView(LocalRow, indices, values);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getLocalDiagCopy(Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &diag) const {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::getLocalDiagCopy(Xpetra::Vector<Scalar,Node> &diag) const {
#endif
    matrixData_->getLocalDiagCopy(diag);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getLocalDiagOffsets(Teuchos::ArrayRCP<size_t> &offsets) const {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::getLocalDiagOffsets(Teuchos::ArrayRCP<size_t> &offsets) const {
#endif
    matrixData_->getLocalDiagOffsets(offsets);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getLocalDiagCopy(Xpetra::Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &diag, const Teuchos::ArrayView<const size_t> &offsets) const {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::getLocalDiagCopy(Xpetra::Vector< Scalar, Node > &diag, const Teuchos::ArrayView<const size_t> &offsets) const {
#endif
    matrixData_->getLocalDiagCopy(diag,offsets);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  typename ScalarTraits<Scalar>::magnitudeType CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getFrobeniusNorm() const {
#else
  template <class Scalar, class Node>
  typename ScalarTraits<Scalar>::magnitudeType CrsMatrixWrap<Scalar,Node>::getFrobeniusNorm() const {
#endif
    return matrixData_->getFrobeniusNorm();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::leftScale (const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& x) {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::leftScale (const Vector<Scalar, Node>& x) {
#endif
    matrixData_->leftScale(x);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::rightScale (const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& x) {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::rightScale (const Vector<Scalar, Node>& x) {
#endif
    matrixData_->rightScale(x);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  bool CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::haveGlobalConstants() const {
#else
  template <class Scalar, class Node>
  bool CrsMatrixWrap<Scalar,Node>::haveGlobalConstants() const {
#endif
    return matrixData_->haveGlobalConstants();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::apply(const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& X,
                   Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Y,
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::apply(const Xpetra::MultiVector<Scalar,Node>& X,
                   Xpetra::MultiVector<Scalar,Node>& Y,
#endif
                   Teuchos::ETransp mode,
                   Scalar alpha,
                   Scalar beta) const {

    matrixData_->apply(X,Y,mode,alpha,beta);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getDomainMap() const {
#else
  template <class Scalar, class Node>
  RCP<const Xpetra::Map<Node> > CrsMatrixWrap<Scalar,Node>::getDomainMap() const {
#endif
    return matrixData_->getDomainMap();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getRangeMap() const {
#else
  template <class Scalar, class Node>
  RCP<const Xpetra::Map<Node> > CrsMatrixWrap<Scalar,Node>::getRangeMap() const {
#endif
    return matrixData_->getRangeMap();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  const RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>> & CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getColMap() const { return getColMap(Matrix::GetCurrentViewLabel()); }
#else
  template <class Scalar, class Node>
  const RCP<const Xpetra::Map<Node>> & CrsMatrixWrap<Scalar,Node>::getColMap() const { return getColMap(Matrix::GetCurrentViewLabel()); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  const RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>> & CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getColMap(viewLabel_t viewLabel) const {
#else
  template <class Scalar, class Node>
  const RCP<const Xpetra::Map<Node>> & CrsMatrixWrap<Scalar,Node>::getColMap(viewLabel_t viewLabel) const {
#endif
    TEUCHOS_TEST_FOR_EXCEPTION(Matrix::operatorViewTable_.containsKey(viewLabel) == false, Xpetra::Exceptions::RuntimeError, "Xpetra::Matrix.GetColMap(): view '" + viewLabel + "' does not exist.");
    updateDefaultView(); // If CrsMatrix::fillComplete() have been used instead of CrsMatrixWrap::fillComplete(), the default view is updated.
    return Matrix::operatorViewTable_.get(viewLabel)->GetColMap();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::removeEmptyProcessesInPlace(const Teuchos::RCP<const Map>& newMap) {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::removeEmptyProcessesInPlace(const Teuchos::RCP<const Map>& newMap) {
#endif
    matrixData_->removeEmptyProcessesInPlace(newMap);
    this->operatorViewTable_.get(this->GetCurrentViewLabel())->SetRowMap(matrixData_->getRowMap());
    this->operatorViewTable_.get(this->GetCurrentViewLabel())->SetColMap(matrixData_->getColMap());
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  const Teuchos::RCP< const Xpetra::Map< LocalOrdinal, GlobalOrdinal, Node > > CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getMap() const {
#else
  template <class Scalar, class Node>
  const Teuchos::RCP< const Xpetra::Map<Node > > CrsMatrixWrap<Scalar,Node>::getMap() const {
#endif
    return matrixData_->getMap();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::doImport(const Matrix &source,
                const Xpetra::Import< LocalOrdinal, GlobalOrdinal, Node > &importer, CombineMode CM) {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::doImport(const Matrix &source,
                const Xpetra::Import<Node > &importer, CombineMode CM) {
#endif
    const CrsMatrixWrap & sourceWrp = dynamic_cast<const CrsMatrixWrap &>(source);
    matrixData_->doImport(*sourceWrp.getCrsMatrix(), importer, CM);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::doExport(const Matrix &dest,
                const Xpetra::Import< LocalOrdinal, GlobalOrdinal, Node >& importer, CombineMode CM) {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::doExport(const Matrix &dest,
                const Xpetra::Import<Node >& importer, CombineMode CM) {
#endif
    const CrsMatrixWrap & destWrp = dynamic_cast<const CrsMatrixWrap &>(dest);
    matrixData_->doExport(*destWrp.getCrsMatrix(), importer, CM);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::doImport(const Matrix &source,
                const Xpetra::Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM) {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::doImport(const Matrix &source,
                const Xpetra::Export<Node >& exporter, CombineMode CM) {
#endif
    const CrsMatrixWrap & sourceWrp = dynamic_cast<const CrsMatrixWrap &>(source);
    matrixData_->doImport(*sourceWrp.getCrsMatrix(), exporter, CM);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::doExport(const Matrix &dest,
                const Xpetra::Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM) {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::doExport(const Matrix &dest,
                const Xpetra::Export<Node >& exporter, CombineMode CM) {
#endif
    const CrsMatrixWrap & destWrp = dynamic_cast<const CrsMatrixWrap &>(dest);
    matrixData_->doExport(*destWrp.getCrsMatrix(), exporter, CM);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  std::string CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::description() const {
#else
  template <class Scalar, class Node>
  std::string CrsMatrixWrap<Scalar,Node>::description() const {
#endif
    return "Xpetra::CrsMatrixWrap";
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel) const {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel) const {
#endif
    //     Teuchos::EVerbosityLevel vl = verbLevel;
    //     if (vl == VERB_DEFAULT) vl = VERB_LOW;
    //     RCP<const Comm<int> > comm = this->getComm();
    //     const int myImageID = comm->getRank(),
    //       numImages = comm->getSize();

    //     if (myImageID == 0) out << this->description() << std::endl;

    matrixData_->describe(out,verbLevel);

    // Teuchos::OSTab tab(out);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::setObjectLabel( const std::string &objectLabel ) {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::setObjectLabel( const std::string &objectLabel ) {
#endif
    Teuchos::LabeledObject::setObjectLabel(objectLabel);
    matrixData_->setObjectLabel(objectLabel);
  }

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef HAVE_XPETRA_TPETRA
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  typename Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type
  CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getLocalMatrix () const {
#else
  template <class Scalar, class Node>
  typename Xpetra::CrsMatrix<Scalar, Node>::local_matrix_type
  CrsMatrixWrap<Scalar,Node>::getLocalMatrix () const {
#endif
    return matrixData_->getLocalMatrix();
  }
#else
#ifdef __GNUC__
#warning "Xpetra Kokkos interface for CrsMatrix is enabled (HAVE_XPETRA_KOKKOS_REFACTOR) but Tpetra is disabled. The Kokkos interface needs Tpetra to be enabled, too."
#endif
#endif
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  bool CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::hasCrsGraph() const {return true;}
#else
  template <class Scalar, class Node>
  bool CrsMatrixWrap<Scalar,Node>::hasCrsGraph() const {return true;}
#endif


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const Xpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node>> CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getCrsGraph() const { return matrixData_->getCrsGraph(); }
#else
  template <class Scalar, class Node>
  RCP<const Xpetra::CrsGraph<Node>> CrsMatrixWrap<Scalar,Node>::getCrsGraph() const { return matrixData_->getCrsGraph(); }
#endif


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getCrsMatrix() const {  return matrixData_; }
#else
  template <class Scalar, class Node>
  RCP<Xpetra::CrsMatrix<Scalar, Node>> CrsMatrixWrap<Scalar,Node>::getCrsMatrix() const {  return matrixData_; }
#endif

// Default view is created after fillComplete()
  // Because ColMap might not be available before fillComplete().
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::CreateDefaultView() {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::CreateDefaultView() {
#endif

    // Create default view
    this->defaultViewLabel_ = "point";
    this->CreateView(this->GetDefaultViewLabel(), matrixData_->getRowMap(), matrixData_->getColMap());

    // Set current view
    this->currentViewLabel_ = this->GetDefaultViewLabel();
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::updateDefaultView() const {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::updateDefaultView() const {
#endif
    if ((finalDefaultView_ == false) &&  matrixData_->isFillComplete() ) {
      // Update default view with the colMap
      Matrix::operatorViewTable_.get(Matrix::GetDefaultViewLabel())->SetColMap(matrixData_->getColMap());
      finalDefaultView_ = true;
    }
  }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>::residual(
            const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > & X, 
            const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > & B,
            MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > & R) const {
#else
  template <class Scalar, class Node>
  void CrsMatrixWrap<Scalar,Node>::residual(
            const MultiVector< Scalar, Node > & X, 
            const MultiVector< Scalar, Node > & B,
            MultiVector< Scalar, Node > & R) const {
#endif
    matrixData_->residual(X,B,R);
  }


} //namespace Xpetra

#endif //ifndef XPETRA_CRSMATRIXWRAP_DEF_HPP
