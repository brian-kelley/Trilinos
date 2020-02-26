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
#ifndef XPETRA_TPETRACRSGRAPH_DEF_HPP
#define XPETRA_TPETRACRSGRAPH_DEF_HPP
#include "Xpetra_TpetraConfigDefs.hpp"
#include "Xpetra_Exceptions.hpp"

#include "Tpetra_CrsGraph.hpp"

#include "Xpetra_CrsGraph.hpp"
#include "Xpetra_TpetraCrsGraph_decl.hpp"
#include "Xpetra_Utils.hpp"
#include "Xpetra_TpetraMap.hpp"
#include "Xpetra_TpetraImport.hpp"
#include "Xpetra_TpetraExport.hpp"


namespace Xpetra {
#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsGraph(const RCP< const map_type > &rowMap, size_t maxNumEntriesPerRow, const RCP< ParameterList > &params)
: graph_(Teuchos::rcp(new Tpetra::CrsGraph< LocalOrdinal, GlobalOrdinal, Node >(toTpetra(rowMap), maxNumEntriesPerRow, Tpetra::StaticProfile, params))) {  }

template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsGraph(const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const RCP< ParameterList > &params)
: graph_(Teuchos::rcp(new Tpetra::CrsGraph< LocalOrdinal, GlobalOrdinal, Node >(toTpetra(rowMap), NumEntriesPerRowToAlloc(), Tpetra::StaticProfile, params))) {  }

template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsGraph(const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, size_t maxNumEntriesPerRow, const RCP< ParameterList > &params)
: graph_(Teuchos::rcp(new Tpetra::CrsGraph< LocalOrdinal, GlobalOrdinal, Node >(toTpetra(rowMap), toTpetra(colMap), maxNumEntriesPerRow, Tpetra::StaticProfile, params))) {  }

template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsGraph(const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const RCP< ParameterList > &params)
: graph_(Teuchos::rcp(new Tpetra::CrsGraph< LocalOrdinal, GlobalOrdinal, Node >(toTpetra(rowMap), toTpetra(colMap), NumEntriesPerRowToAlloc(), Tpetra::StaticProfile, params))) {  }
#else
template<class Node>
TpetraCrsGraph<Node>::TpetraCrsGraph(const RCP< const map_type > &rowMap, size_t maxNumEntriesPerRow, const RCP< ParameterList > &params)
: graph_(Teuchos::rcp(new Tpetra::CrsGraph<Node >(toTpetra(rowMap), maxNumEntriesPerRow, Tpetra::StaticProfile, params))) {  }

template<class Node>
TpetraCrsGraph<Node>::TpetraCrsGraph(const RCP< const Map<Node > > &rowMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const RCP< ParameterList > &params)
: graph_(Teuchos::rcp(new Tpetra::CrsGraph<Node >(toTpetra(rowMap), NumEntriesPerRowToAlloc(), Tpetra::StaticProfile, params))) {  }

template<class Node>
TpetraCrsGraph<Node>::TpetraCrsGraph(const RCP< const Map<Node > > &rowMap, const RCP< const Map<Node > > &colMap, size_t maxNumEntriesPerRow, const RCP< ParameterList > &params)
: graph_(Teuchos::rcp(new Tpetra::CrsGraph<Node >(toTpetra(rowMap), toTpetra(colMap), maxNumEntriesPerRow, Tpetra::StaticProfile, params))) {  }

template<class Node>
TpetraCrsGraph<Node>::TpetraCrsGraph(const RCP< const Map<Node > > &rowMap, const RCP< const Map<Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const RCP< ParameterList > &params)
: graph_(Teuchos::rcp(new Tpetra::CrsGraph<Node >(toTpetra(rowMap), toTpetra(colMap), NumEntriesPerRowToAlloc(), Tpetra::StaticProfile, params))) {  }
#endif

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::
TpetraCrsGraph(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap,
               const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap,
#else
template<class Node>
TpetraCrsGraph<Node>::
TpetraCrsGraph(const Teuchos::RCP< const Map<Node > > &rowMap,
               const Teuchos::RCP< const Map<Node > > &colMap,
#endif
               const typename local_graph_type::row_map_type& rowPointers,
               const typename local_graph_type::entries_type::non_const_type& columnIndices,
               const Teuchos::RCP< Teuchos::ParameterList > &plist)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  : graph_(Teuchos::rcp(new Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node>(toTpetra(rowMap), toTpetra(colMap), rowPointers, columnIndices, plist))) {  }
#else
  : graph_(Teuchos::rcp(new Tpetra::CrsGraph<Node>(toTpetra(rowMap), toTpetra(colMap), rowPointers, columnIndices, plist))) {  }
#endif
  
  
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::
#else
template<class Node>
TpetraCrsGraph<Node>::
#endif
TpetraCrsGraph(const Teuchos::RCP<const map_type>& rowMap,
               const Teuchos::RCP<const map_type>& colMap,
               const local_graph_type& lclGraph,
               const Teuchos::RCP<Teuchos::ParameterList>& params)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  : graph_(Teuchos::rcp(new Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node>(toTpetra(rowMap), toTpetra(colMap), lclGraph, params))) {  }
#else
  : graph_(Teuchos::rcp(new Tpetra::CrsGraph<Node>(toTpetra(rowMap), toTpetra(colMap), lclGraph, params))) {  }
#endif
  
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::
#else
template<class Node>
TpetraCrsGraph<Node>::
#endif
TpetraCrsGraph(const local_graph_type& lclGraph,
               const Teuchos::RCP<const map_type>& rowMap,
               const Teuchos::RCP<const map_type>& colMap,
               const Teuchos::RCP<const map_type>& domainMap,
               const Teuchos::RCP<const map_type>& rangeMap,
               const Teuchos::RCP<Teuchos::ParameterList>& params)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  : graph_(Teuchos::rcp(new Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node>(lclGraph, toTpetra(rowMap), toTpetra(colMap), toTpetra(domainMap), toTpetra(rangeMap), params))) {  }
#else
  : graph_(Teuchos::rcp(new Tpetra::CrsGraph<Node>(lclGraph, toTpetra(rowMap), toTpetra(colMap), toTpetra(domainMap), toTpetra(rangeMap), params))) {  }
#endif
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
 TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::~TpetraCrsGraph() {  }
#else
template<class Node>
 TpetraCrsGraph<Node>::~TpetraCrsGraph() {  }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::insertGlobalIndices(GlobalOrdinal globalRow, const ArrayView< const GlobalOrdinal > &indices)
#else
template<class Node>
void TpetraCrsGraph<Node>::insertGlobalIndices(GlobalOrdinal globalRow, const ArrayView< const GlobalOrdinal > &indices)
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::insertGlobalIndices"); graph_->insertGlobalIndices(globalRow, indices); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::insertLocalIndices(const LocalOrdinal localRow, const ArrayView< const LocalOrdinal > &indices)
#else
template<class Node>
void TpetraCrsGraph<Node>::insertLocalIndices(const LocalOrdinal localRow, const ArrayView< const LocalOrdinal > &indices)
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::insertLocalIndices"); graph_->insertLocalIndices(localRow, indices); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::removeLocalIndices(LocalOrdinal localRow)
#else
template<class Node>
void TpetraCrsGraph<Node>::removeLocalIndices(LocalOrdinal localRow)
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::removeLocalIndices"); graph_->removeLocalIndices(localRow); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::fillComplete(const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &domainMap, const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rangeMap, const RCP< ParameterList > &params)
#else
template<class Node>
void TpetraCrsGraph<Node>::fillComplete(const RCP< const Map<Node > > &domainMap, const RCP< const Map<Node > > &rangeMap, const RCP< ParameterList > &params)
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::fillComplete"); graph_->fillComplete(toTpetra(domainMap), toTpetra(rangeMap), params); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::fillComplete(const RCP< ParameterList > &params)
#else
template<class Node>
void TpetraCrsGraph<Node>::fillComplete(const RCP< ParameterList > &params)
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::fillComplete"); graph_->fillComplete(params); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
RCP< const Comm< int > > TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getComm() const
#else
template<class Node>
RCP< const Comm< int > > TpetraCrsGraph<Node>::getComm() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getComm"); return graph_->getComm(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getRowMap() const
#else
template<class Node>
RCP< const Map<Node > >  TpetraCrsGraph<Node>::getRowMap() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getRowMap"); return toXpetra(graph_->getRowMap()); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getColMap() const
#else
template<class Node>
RCP< const Map<Node > >  TpetraCrsGraph<Node>::getColMap() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getColMap"); return toXpetra(graph_->getColMap()); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getDomainMap() const
#else
template<class Node>
RCP< const Map<Node > >  TpetraCrsGraph<Node>::getDomainMap() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getDomainMap"); return toXpetra(graph_->getDomainMap()); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getRangeMap() const
#else
template<class Node>
RCP< const Map<Node > >  TpetraCrsGraph<Node>::getRangeMap() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getRangeMap"); return toXpetra(graph_->getRangeMap()); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
RCP< const Import< LocalOrdinal, GlobalOrdinal, Node > > TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getImporter() const
#else
template<class Node>
RCP< const Import<Node > > TpetraCrsGraph<Node>::getImporter() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getImporter"); return toXpetra(graph_->getImporter()); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
RCP< const Export< LocalOrdinal, GlobalOrdinal, Node > > TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getExporter() const
#else
template<class Node>
RCP< const Export<Node > > TpetraCrsGraph<Node>::getExporter() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getExporter"); return toXpetra(graph_->getExporter()); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
global_size_t TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getGlobalNumRows() const
#else
template<class Node>
global_size_t TpetraCrsGraph<Node>::getGlobalNumRows() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getGlobalNumRows"); return graph_->getGlobalNumRows(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
global_size_t TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getGlobalNumCols() const
#else
template<class Node>
global_size_t TpetraCrsGraph<Node>::getGlobalNumCols() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getGlobalNumCols"); return graph_->getGlobalNumCols(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
size_t TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getNodeNumRows() const
#else
template<class Node>
size_t TpetraCrsGraph<Node>::getNodeNumRows() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getNodeNumRows"); return graph_->getNodeNumRows(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
size_t TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getNodeNumCols() const
#else
template<class Node>
size_t TpetraCrsGraph<Node>::getNodeNumCols() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getNodeNumCols"); return graph_->getNodeNumCols(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
GlobalOrdinal TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getIndexBase() const
#else
template<class Node>
GlobalOrdinal TpetraCrsGraph<Node>::getIndexBase() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getIndexBase"); return graph_->getIndexBase(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
global_size_t TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getGlobalNumEntries() const
#else
template<class Node>
global_size_t TpetraCrsGraph<Node>::getGlobalNumEntries() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getGlobalNumEntries"); return graph_->getGlobalNumEntries(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
size_t TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getNodeNumEntries() const
#else
template<class Node>
size_t TpetraCrsGraph<Node>::getNodeNumEntries() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getNodeNumEntries"); return graph_->getNodeNumEntries(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
size_t TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getNumEntriesInGlobalRow(GlobalOrdinal globalRow) const
#else
template<class Node>
size_t TpetraCrsGraph<Node>::getNumEntriesInGlobalRow(GlobalOrdinal globalRow) const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getNumEntriesInGlobalRow"); return graph_->getNumEntriesInGlobalRow(globalRow); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
size_t TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getNumEntriesInLocalRow(LocalOrdinal localRow) const
#else
template<class Node>
size_t TpetraCrsGraph<Node>::getNumEntriesInLocalRow(LocalOrdinal localRow) const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getNumEntriesInLocalRow"); return graph_->getNumEntriesInLocalRow(localRow); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
size_t TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getNumAllocatedEntriesInGlobalRow(GlobalOrdinal globalRow) const
#else
template<class Node>
size_t TpetraCrsGraph<Node>::getNumAllocatedEntriesInGlobalRow(GlobalOrdinal globalRow) const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getNumAllocatedEntriesInGlobalRow"); return graph_->getNumAllocatedEntriesInGlobalRow(globalRow); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
size_t TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getNumAllocatedEntriesInLocalRow(LocalOrdinal localRow) const
#else
template<class Node>
size_t TpetraCrsGraph<Node>::getNumAllocatedEntriesInLocalRow(LocalOrdinal localRow) const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getNumAllocatedEntriesInLocalRow"); return graph_->getNumAllocatedEntriesInLocalRow(localRow); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
size_t TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getGlobalMaxNumRowEntries() const
#else
template<class Node>
size_t TpetraCrsGraph<Node>::getGlobalMaxNumRowEntries() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getGlobalMaxNumRowEntries"); return graph_->getGlobalMaxNumRowEntries(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
size_t TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getNodeMaxNumRowEntries() const
#else
template<class Node>
size_t TpetraCrsGraph<Node>::getNodeMaxNumRowEntries() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getNodeMaxNumRowEntries"); return graph_->getNodeMaxNumRowEntries(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
bool TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::hasColMap() const
#else
template<class Node>
bool TpetraCrsGraph<Node>::hasColMap() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::hasColMap"); return graph_->hasColMap(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
bool TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::isLocallyIndexed() const
#else
template<class Node>
bool TpetraCrsGraph<Node>::isLocallyIndexed() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::isLocallyIndexed"); return graph_->isLocallyIndexed(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
bool TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::isGloballyIndexed() const
#else
template<class Node>
bool TpetraCrsGraph<Node>::isGloballyIndexed() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::isGloballyIndexed"); return graph_->isGloballyIndexed(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
bool TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::isFillComplete() const
#else
template<class Node>
bool TpetraCrsGraph<Node>::isFillComplete() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::isFillComplete"); return graph_->isFillComplete(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
bool TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::isStorageOptimized() const
#else
template<class Node>
bool TpetraCrsGraph<Node>::isStorageOptimized() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::isStorageOptimized"); return graph_->isStorageOptimized(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getGlobalRowView(GlobalOrdinal GlobalRow, ArrayView< const GlobalOrdinal > &Indices) const
#else
template<class Node>
void TpetraCrsGraph<Node>::getGlobalRowView(GlobalOrdinal GlobalRow, ArrayView< const GlobalOrdinal > &Indices) const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getGlobalRowView"); graph_->getGlobalRowView(GlobalRow, Indices); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getLocalRowView(LocalOrdinal LocalRow, ArrayView< const LocalOrdinal > &indices) const
#else
template<class Node>
void TpetraCrsGraph<Node>::getLocalRowView(LocalOrdinal LocalRow, ArrayView< const LocalOrdinal > &indices) const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getLocalRowView"); graph_->getLocalRowView(LocalRow, indices); }

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
typename Xpetra::CrsGraph<LocalOrdinal,GlobalOrdinal,Node>::local_graph_type TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getLocalGraph () const {
#else
template<class Node>
typename Xpetra::CrsGraph<Node>::local_graph_type TpetraCrsGraph<Node>::getLocalGraph () const {
#endif
  return getTpetra_CrsGraph()->getLocalGraph();
}
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::computeGlobalConstants() {
#else
template<class Node>
void TpetraCrsGraph<Node>::computeGlobalConstants() {
#endif
      // mfh 07 May 2018: See GitHub Issue #2565.
      constexpr bool computeLocalTriangularConstants = true;
      graph_->computeGlobalConstants(computeLocalTriangularConstants);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
std::string TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::description() const
#else
template<class Node>
std::string TpetraCrsGraph<Node>::description() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::description"); return graph_->description(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel) const
#else
template<class Node>
void TpetraCrsGraph<Node>::describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel) const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::describe"); graph_->describe(out, verbLevel); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
ArrayRCP< const size_t > TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getNodeRowPtrs() const
#else
template<class Node>
ArrayRCP< const size_t > TpetraCrsGraph<Node>::getNodeRowPtrs() const
#endif
{ XPETRA_MONITOR("TpetraCrsGraph::getNodeRowPtrs"); return graph_->getNodeRowPtrs(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getMap() const
{ XPETRA_MONITOR("TpetraCrsGraph::getMap"); return rcp( new TpetraMap< LocalOrdinal, GlobalOrdinal, Node >(graph_->getMap()) ); }

template<class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::doImport(const DistObject<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> &source,
                                                               const Import< LocalOrdinal, GlobalOrdinal, Node > &importer, CombineMode CM){
#else
template<class Node>
Teuchos::RCP< const Map<Node > > TpetraCrsGraph<Node>::getMap() const
{ XPETRA_MONITOR("TpetraCrsGraph::getMap"); return rcp( new TpetraMap<Node >(graph_->getMap()) ); }

template<class Node>
void TpetraCrsGraph<Node>::doImport(const DistObject<GlobalOrdinal, Node> &source,
                                                               const Import<Node > &importer, CombineMode CM){
#endif
  XPETRA_MONITOR("TpetraCrsGraph::doImport");
  
  XPETRA_DYNAMIC_CAST(const TpetraCrsGraphClass, source, tSource, "Xpetra::TpetraCrsGraph::doImport only accept Xpetra::TpetraCrsGraph as input arguments.");//TODO: remove and use toTpetra()
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  RCP< const Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node> > v = tSource.getTpetra_CrsGraph();
#else
  RCP< const Tpetra::CrsGraph<Node> > v = tSource.getTpetra_CrsGraph();
#endif
  //graph_->doImport(toTpetraCrsGraph(source), *tImporter.getTpetra_Import(), toTpetra(CM));
  
  graph_->doImport(*v, toTpetra(importer), toTpetra(CM));
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::doExport(const DistObject<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> &dest,
                                                               const Import< LocalOrdinal, GlobalOrdinal, Node >& importer, CombineMode CM) {
#else
template<class Node>
void TpetraCrsGraph<Node>::doExport(const DistObject<GlobalOrdinal, Node> &dest,
                                                               const Import<Node >& importer, CombineMode CM) {
#endif
  XPETRA_MONITOR("TpetraCrsGraph::doExport");
  
  XPETRA_DYNAMIC_CAST(const TpetraCrsGraphClass, dest, tDest, "Xpetra::TpetraCrsGraph::doImport only accept Xpetra::TpetraCrsGraph as input arguments.");//TODO: remove and use toTpetra()
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  RCP< const Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node> > v = tDest.getTpetra_CrsGraph();
#else
  RCP< const Tpetra::CrsGraph<Node> > v = tDest.getTpetra_CrsGraph();
#endif
  graph_->doExport(*v, toTpetra(importer), toTpetra(CM));
  
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::doImport(const DistObject<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> &source,
                                                               const Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM){
#else
template<class Node>
void TpetraCrsGraph<Node>::doImport(const DistObject<GlobalOrdinal, Node> &source,
                                                               const Export<Node >& exporter, CombineMode CM){
#endif
  XPETRA_MONITOR("TpetraCrsGraph::doImport");
  
  XPETRA_DYNAMIC_CAST(const TpetraCrsGraphClass, source, tSource, "Xpetra::TpetraCrsGraph::doImport only accept Xpetra::TpetraCrsGraph as input arguments.");//TODO: remove and use toTpetra()
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  RCP< const Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node> > v = tSource.getTpetra_CrsGraph();
#else
  RCP< const Tpetra::CrsGraph<Node> > v = tSource.getTpetra_CrsGraph();
#endif
  
  graph_->doImport(*v, toTpetra(exporter), toTpetra(CM));
  
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::doExport(const DistObject<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> &dest,
                                                               const Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM) {
#else
template<class Node>
void TpetraCrsGraph<Node>::doExport(const DistObject<GlobalOrdinal, Node> &dest,
                                                               const Export<Node >& exporter, CombineMode CM) {
#endif
  XPETRA_MONITOR("TpetraCrsGraph::doExport");
  
  XPETRA_DYNAMIC_CAST(const TpetraCrsGraphClass, dest, tDest, "Xpetra::TpetraCrsGraph::doImport only accept Xpetra::TpetraCrsGraph as input arguments.");//TODO: remove and use toTpetra()
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  RCP< const Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node> > v = tDest.getTpetra_CrsGraph();
#else
  RCP< const Tpetra::CrsGraph<Node> > v = tDest.getTpetra_CrsGraph();
#endif
  
      graph_->doExport(*v, toTpetra(exporter), toTpetra(CM));
      
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsGraph(const Teuchos::RCP<Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node> > &graph) : graph_(graph)
#else
template<class Node>
TpetraCrsGraph<Node>::TpetraCrsGraph(const Teuchos::RCP<Tpetra::CrsGraph<Node> > &graph) : graph_(graph)
#endif
{ }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
RCP< const Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node> > TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node>::getTpetra_CrsGraph() const
#else
template<class Node>
RCP< const Tpetra::CrsGraph<Node> > TpetraCrsGraph<Node>::getTpetra_CrsGraph() const
#endif
{ return graph_; }


#ifdef HAVE_XPETRA_EPETRA

#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))

  // specialization of TpetraCrsGraph for GO=LO=int
  template <>
  class TpetraCrsGraph<int,int,EpetraNode>
    : public CrsGraph<int,int,EpetraNode>
  {
    typedef int LocalOrdinal;
    typedef int GlobalOrdinal;
    typedef EpetraNode Node;

    // The following typedef is used by the XPETRA_DYNAMIC_CAST() macro.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node> TpetraCrsGraphClass;
    typedef Map<LocalOrdinal,GlobalOrdinal,Node> map_type;
#else
    typedef TpetraCrsGraph<Node> TpetraCrsGraphClass;
    typedef Map<Node> map_type;
#endif

  public:

    //! @name Constructor/Destructor Methods
    //@{

    //! Constructor specifying fixed number of entries for each row.
    TpetraCrsGraph(const RCP< const map_type > &rowMap, size_t maxNumEntriesPerRow, const RCP< ParameterList > &params=null) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Constructor specifying (possibly different) number of entries in each row.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraCrsGraph(const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const RCP< ParameterList > &params=null) {
#else
    TpetraCrsGraph(const RCP< const Map<Node > > &rowMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const RCP< ParameterList > &params=null) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Constructor specifying column Map and fixed number of entries for each row.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraCrsGraph(const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, size_t maxNumEntriesPerRow, const RCP< ParameterList > &params=null) {
#else
    TpetraCrsGraph(const RCP< const Map<Node > > &rowMap, const RCP< const Map<Node > > &colMap, size_t maxNumEntriesPerRow, const RCP< ParameterList > &params=null) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Constructor specifying column Map and number of entries in each row.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraCrsGraph(const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const RCP< ParameterList > &params=null) {
#else
    TpetraCrsGraph(const RCP< const Map<Node > > &rowMap, const RCP< const Map<Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const RCP< ParameterList > &params=null) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
    /// \brief Constructor specifying column Map and arrays containing the graph in sorted, local ids.
    ///
    ///
    /// \param rowMap [in] Distribution of rows of the graph.
    ///
    /// \param colMap [in] Distribution of columns of the graph.
    ///
    /// \param rowPointers [in] The beginning of each row in the graph,
    ///   as in a CSR "rowptr" array.  The length of this vector should be
    ///   equal to the number of rows in the graph, plus one.  This last
    ///   entry should store the nunber of nonzeros in the graph.
    ///
    /// \param columnIndices [in] The local indices of the columns,
    ///   as in a CSR "colind" array.  The length of this vector
    ///   should be equal to the number of unknowns in the graph.
    ///
    /// \param params [in/out] Optional list of parameters.  If not
    ///   null, any missing parameters will be filled in with their
    ///   default values.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraCrsGraph(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap,
                   const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap,
#else
    TpetraCrsGraph(const Teuchos::RCP< const Map<Node > > &rowMap,
                   const Teuchos::RCP< const Map<Node > > &colMap,
#endif
                   const typename local_graph_type::row_map_type& rowPointers,
                   const typename local_graph_type::entries_type::non_const_type& columnIndices,
                   const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(),
                                   typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(),
                                   "int",
                                   typeid(EpetraNode).name());
    }

    /// \brief Constructor specifying column Map and a local (sorted)
    ///   graph, which the resulting CrsGraph views.
    ///
    /// Unlike most other CrsGraph constructors, successful completion
    /// of this constructor will result in a fill-complete graph.
    ///
    /// \param rowMap [in] Distribution of rows of the graph.
    ///
    /// \param colMap [in] Distribution of columns of the graph.
    ///
    /// \param lclGraph [in] A locally indexed Kokkos::StaticCrsGraph
    ///   whose local row indices come from the specified row Map, and
    ///   whose local column indices come from the specified column
    ///   Map.
    ///
    /// \param params [in/out] Optional list of parameters.  If not
    ///   null, any missing parameters will be filled in with their
    ///   default values.
    TpetraCrsGraph(const Teuchos::RCP<const map_type>& rowMap,
                   const Teuchos::RCP<const map_type>& colMap,
                   const local_graph_type& lclGraph,
                   const Teuchos::RCP<Teuchos::ParameterList>& params) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(),
                                   typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(),
                                   "int",
                                   typeid(EpetraNode).name());
    }

    /// \brief Constructor specifying column, domain and range maps, and a
    ///   local (sorted) graph, which the resulting CrsGraph views.
    ///
    /// Unlike most other CrsGraph constructors, successful completion
    /// of this constructor will result in a fill-complete graph.
    ///
    /// \param rowMap [in] Distribution of rows of the graph.
    ///
    /// \param colMap [in] Distribution of columns of the graph.
    ///
    /// \param domainMap [in] The graph's domain Map. MUST be one to
    ///   one!
    ///
    /// \param rangeMap [in] The graph's range Map.  MUST be one to
    ///   one!  May be, but need not be, the same as the domain Map.
    ///
    /// \param lclGraph [in] A locally indexed Kokkos::StaticCrsGraph
    ///   whose local row indices come from the specified row Map, and
    ///   whose local column indices come from the specified column
    ///   Map.
    ///
    /// \param params [in/out] Optional list of parameters.  If not
    ///   null, any missing parameters will be filled in with their
    ///   default values.
    TpetraCrsGraph(const local_graph_type& lclGraph,
                   const Teuchos::RCP<const map_type>& rowMap,
                   const Teuchos::RCP<const map_type>& colMap,
                   const Teuchos::RCP<const map_type>& domainMap = Teuchos::null,
                   const Teuchos::RCP<const map_type>& rangeMap = Teuchos::null,
                   const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(),
                                   typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(),
                                   "int",
                                   typeid(EpetraNode).name());
    }
#endif

    //! Destructor.
    virtual ~TpetraCrsGraph() {  }

    //@}

    //! @name Insertion/Removal Methods
    //@{

    //! Insert global indices into the graph.
    void insertGlobalIndices(GlobalOrdinal globalRow, const ArrayView< const GlobalOrdinal > &indices) { }

    //! Insert local indices into the graph.
    void insertLocalIndices(const LocalOrdinal localRow, const ArrayView< const LocalOrdinal > &indices) { }

    //! Remove all graph indices from the specified local row.
    void removeLocalIndices(LocalOrdinal localRow) { }

    //@}

    //! @name Transformational Methods
    //@{

    //! Signal that data entry is complete, specifying domain and range maps.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void fillComplete(const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &domainMap, const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rangeMap, const RCP< ParameterList > &params=null) { }
#else
    void fillComplete(const RCP< const Map<Node > > &domainMap, const RCP< const Map<Node > > &rangeMap, const RCP< ParameterList > &params=null) { }
#endif

    //! Signal that data entry is complete.
    void fillComplete(const RCP< ParameterList > &params=null) { }

    //@}

    //! @name Methods implementing RowGraph.
    //@{

    //! Returns the communicator.
    RCP< const Comm< int > > getComm() const { return Teuchos::null; }

    //! Returns the Map that describes the row distribution in this graph.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  getRowMap() const { return Teuchos::null; }
#else
    RCP< const Map<Node > >  getRowMap() const { return Teuchos::null; }
#endif

    //! Returns the Map that describes the column distribution in this graph.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  getColMap() const { return Teuchos::null; }
#else
    RCP< const Map<Node > >  getColMap() const { return Teuchos::null; }
#endif

    //! Returns the Map associated with the domain of this graph.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  getDomainMap() const { return Teuchos::null; }
#else
    RCP< const Map<Node > >  getDomainMap() const { return Teuchos::null; }
#endif

    //! Returns the Map associated with the domain of this graph.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  getRangeMap() const { return Teuchos::null; }
#else
    RCP< const Map<Node > >  getRangeMap() const { return Teuchos::null; }
#endif

    //! Returns the importer associated with this graph.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Import< LocalOrdinal, GlobalOrdinal, Node > > getImporter() const { return Teuchos::null; }
#else
    RCP< const Import<Node > > getImporter() const { return Teuchos::null; }
#endif

    //! Returns the exporter associated with this graph.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Export< LocalOrdinal, GlobalOrdinal, Node > > getExporter() const { return Teuchos::null; }
#else
    RCP< const Export<Node > > getExporter() const { return Teuchos::null; }
#endif

    //! Returns the number of global rows in the graph.
    global_size_t getGlobalNumRows() const { return 0; }

    //! Returns the number of global columns in the graph.
    global_size_t getGlobalNumCols() const { return 0; }

    //! Returns the number of graph rows owned on the calling node.
    size_t getNodeNumRows() const { return 0; }

    //! Returns the number of columns connected to the locally owned rows of this graph.
    size_t getNodeNumCols() const { return 0; }

    //! Returns the index base for global indices for this graph.
    GlobalOrdinal getIndexBase() const { return 0; }

    //! Returns the global number of entries in the graph.
    global_size_t getGlobalNumEntries() const { return 0; }

    //! Returns the local number of entries in the graph.
    size_t getNodeNumEntries() const { return 0; }

    //! Returns the current number of entries on this node in the specified global row.
    size_t getNumEntriesInGlobalRow(GlobalOrdinal globalRow) const { return 0; }

    //! Returns the current number of entries on this node in the specified local row.
    size_t getNumEntriesInLocalRow(LocalOrdinal localRow) const { return 0; }

    //! Returns the current number of allocated entries for this node in the specified global row .
    size_t getNumAllocatedEntriesInGlobalRow(GlobalOrdinal globalRow) const { return 0; }

    //! Returns the current number of allocated entries on this node in the specified local row.
    size_t getNumAllocatedEntriesInLocalRow(LocalOrdinal localRow) const { return 0; }

    //! Maximum number of entries in all rows over all processes.
    size_t getGlobalMaxNumRowEntries() const { return 0; }

    //! Maximum number of entries in all rows owned by the calling process.
    size_t getNodeMaxNumRowEntries() const { return 0; }

    //! Whether the graph has a column Map.
    bool hasColMap() const { return false; }

    //! Whether column indices are stored using local indices on the calling process.
    bool isLocallyIndexed() const { return false; }

    //! Whether column indices are stored using global indices on the calling process.
    bool isGloballyIndexed() const { return false; }

    //! Whether fillComplete() has been called and the graph is in compute mode.
    bool isFillComplete() const { return false; }

    //! Returns true if storage has been optimized.
    bool isStorageOptimized() const { return false; }

    //! Return a const, nonpersisting view of global indices in the given row.
    void getGlobalRowView(GlobalOrdinal GlobalRow, ArrayView< const GlobalOrdinal > &Indices) const {  }

    //! Return a const, nonpersisting view of local indices in the given row.
    void getLocalRowView(LocalOrdinal LocalRow, ArrayView< const LocalOrdinal > &indices) const {  }

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
    /// \brief Access the local KokkosSparse::StaticCrsGraph data
    local_graph_type getLocalGraph () const {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Xpetra::Exceptions::RuntimeError,
                                 "Epetra does not support Kokkos::StaticCrsGraph!");
      TEUCHOS_UNREACHABLE_RETURN((Kokkos::StaticCrsGraph<LocalOrdinal, Kokkos::LayoutLeft, execution_space>()));
    }
#endif

    //! Dummy implementation for computeGlobalConstants
    void computeGlobalConstants() { }

    //@}

    //! @name Overridden from Teuchos::Describable
    //@{

    //! Return a simple one-line description of this object.
    std::string description() const { return std::string(""); }

    //! Print the object with some verbosity level to an FancyOStream object.
    void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel=Teuchos::Describable::verbLevel_default) const {  }

    //@}

    //! @name Advanced methods, at increased risk of deprecation.
    //@{

    //! Get an ArrayRCP of the row-offsets.
    ArrayRCP< const size_t > getNodeRowPtrs() const { return Teuchos::ArrayRCP< const size_t>(); }

    //@}

    //! Implements DistObject interface
    //{@

    //! Access function for the Tpetra::Map this DistObject was constructed with.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > getMap() const { return Teuchos::null; }
#else
    Teuchos::RCP< const Map<Node > > getMap() const { return Teuchos::null; }
#endif

    //! Import.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doImport(const DistObject<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> &source,
                  const Import< LocalOrdinal, GlobalOrdinal, Node > &importer, CombineMode CM) { }
#else
    void doImport(const DistObject<GlobalOrdinal, Node> &source,
                  const Import<Node > &importer, CombineMode CM) { }
#endif

    //! Export.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doExport(const DistObject<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> &dest,
                  const Import< LocalOrdinal, GlobalOrdinal, Node >& importer, CombineMode CM) { }
#else
    void doExport(const DistObject<GlobalOrdinal, Node> &dest,
                  const Import<Node >& importer, CombineMode CM) { }
#endif

    //! Import (using an Exporter).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doImport(const DistObject<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> &source,
                  const Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM) { }
#else
    void doImport(const DistObject<GlobalOrdinal, Node> &source,
                  const Export<Node >& exporter, CombineMode CM) { }
#endif

    //! Export (using an Importer).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doExport(const DistObject<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> &dest,
                  const Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM) { }
#else
    void doExport(const DistObject<GlobalOrdinal, Node> &dest,
                  const Export<Node >& exporter, CombineMode CM) { }
#endif

    // @}

    //! @name Xpetra specific
    //@{

    //! TpetraCrsGraph constructor to wrap a Tpetra::CrsGraph object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraCrsGraph(const Teuchos::RCP<Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node> > &graph)  {
#else
    TpetraCrsGraph(const Teuchos::RCP<Tpetra::CrsGraph<Node> > &graph)  {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Get the underlying Tpetra graph
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node> > getTpetra_CrsGraph() const { return Teuchos::null; }
#else
    RCP< const Tpetra::CrsGraph<Node> > getTpetra_CrsGraph() const { return Teuchos::null; }
#endif

    //@}
  }; // TpetraCrsGraph class (specialization for LO=GO=int and NO=EpetraNode)
#endif

#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_LONG_LONG))))

  // specialization of TpetraCrsGraph for GO=long long and NO=EpetraNode
  template <>
  class TpetraCrsGraph<int,long long,EpetraNode>
    : public CrsGraph<int,long long,EpetraNode>
  {
    typedef int LocalOrdinal;
    typedef long long GlobalOrdinal;
    typedef EpetraNode Node;

    // The following typedef is used by the XPETRA_DYNAMIC_CAST() macro.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,Node> TpetraCrsGraphClass;
    typedef Map<LocalOrdinal,GlobalOrdinal,Node> map_type;
#else
    typedef TpetraCrsGraph<Node> TpetraCrsGraphClass;
    typedef Map<Node> map_type;
#endif

  public:

    //! @name Constructor/Destructor Methods
    //@{

    //! Constructor specifying fixed number of entries for each row.
    TpetraCrsGraph(const RCP< const map_type > &rowMap, size_t maxNumEntriesPerRow, const RCP< ParameterList > &params=null) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

    //! Constructor specifying (possibly different) number of entries in each row.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraCrsGraph(const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const RCP< ParameterList > &params=null) {
#else
    TpetraCrsGraph(const RCP< const Map<Node > > &rowMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const RCP< ParameterList > &params=null) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

    //! Constructor specifying column Map and fixed number of entries for each row.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraCrsGraph(const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, size_t maxNumEntriesPerRow, const RCP< ParameterList > &params=null) {
#else
    TpetraCrsGraph(const RCP< const Map<Node > > &rowMap, const RCP< const Map<Node > > &colMap, size_t maxNumEntriesPerRow, const RCP< ParameterList > &params=null) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

    //! Constructor specifying column Map and number of entries in each row.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraCrsGraph(const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const RCP< ParameterList > &params=null) {
#else
    TpetraCrsGraph(const RCP< const Map<Node > > &rowMap, const RCP< const Map<Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const RCP< ParameterList > &params=null) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
    /// \brief Constructor specifying column Map and arrays containing the graph in sorted, local ids.
    ///
    ///
    /// \param rowMap [in] Distribution of rows of the graph.
    ///
    /// \param colMap [in] Distribution of columns of the graph.
    ///
    /// \param rowPointers [in] The beginning of each row in the graph,
    ///   as in a CSR "rowptr" array.  The length of this vector should be
    ///   equal to the number of rows in the graph, plus one.  This last
    ///   entry should store the nunber of nonzeros in the graph.
    ///
    /// \param columnIndices [in] The local indices of the columns,
    ///   as in a CSR "colind" array.  The length of this vector
    ///   should be equal to the number of unknowns in the graph.
    ///
    /// \param params [in/out] Optional list of parameters.  If not
    ///   null, any missing parameters will be filled in with their
    ///   default values.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraCrsGraph(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap,
                   const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap,
#else
    TpetraCrsGraph(const Teuchos::RCP< const Map<Node > > &rowMap,
                   const Teuchos::RCP< const Map<Node > > &colMap,
#endif
                   const typename local_graph_type::row_map_type& rowPointers,
                   const typename local_graph_type::entries_type::non_const_type& columnIndices,
                   const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(),
                                   typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(),
                                   "int",
                                   typeid(EpetraNode).name());
    }

    /// \brief Constructor specifying column Map and a local (sorted)
    ///   graph, which the resulting CrsGraph views.
    ///
    /// Unlike most other CrsGraph constructors, successful completion
    /// of this constructor will result in a fill-complete graph.
    ///
    /// \param rowMap [in] Distribution of rows of the graph.
    ///
    /// \param colMap [in] Distribution of columns of the graph.
    ///
    /// \param lclGraph [in] A locally indexed Kokkos::StaticCrsGraph
    ///   whose local row indices come from the specified row Map, and
    ///   whose local column indices come from the specified column
    ///   Map.
    ///
    /// \param params [in/out] Optional list of parameters.  If not
    ///   null, any missing parameters will be filled in with their
    ///   default values.
    TpetraCrsGraph(const Teuchos::RCP<const map_type>& rowMap,
                   const Teuchos::RCP<const map_type>& colMap,
                   const local_graph_type& lclGraph,
                   const Teuchos::RCP<Teuchos::ParameterList>& params) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(),
                                   typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(),
                                   "int",
                                   typeid(EpetraNode).name());
    }

    /// \brief Constructor specifying column, domain and range maps, and a
    ///   local (sorted) graph, which the resulting CrsGraph views.
    ///
    /// Unlike most other CrsGraph constructors, successful completion
    /// of this constructor will result in a fill-complete graph.
    ///
    /// \param rowMap [in] Distribution of rows of the graph.
    ///
    /// \param colMap [in] Distribution of columns of the graph.
    ///
    /// \param domainMap [in] The graph's domain Map. MUST be one to
    ///   one!
    ///
    /// \param rangeMap [in] The graph's range Map.  MUST be one to
    ///   one!  May be, but need not be, the same as the domain Map.
    ///
    /// \param lclGraph [in] A locally indexed Kokkos::StaticCrsGraph
    ///   whose local row indices come from the specified row Map, and
    ///   whose local column indices come from the specified column
    ///   Map.
    ///
    /// \param params [in/out] Optional list of parameters.  If not
    ///   null, any missing parameters will be filled in with their
    ///   default values.
    TpetraCrsGraph(const local_graph_type& lclGraph,
                   const Teuchos::RCP<const map_type>& rowMap,
                   const Teuchos::RCP<const map_type>& colMap,
                   const Teuchos::RCP<const map_type>& domainMap = Teuchos::null,
                   const Teuchos::RCP<const map_type>& rangeMap = Teuchos::null,
                   const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(),
                                   typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(),
                                   "int",
                                   typeid(EpetraNode).name());
    }
#endif

    //! Destructor.
    virtual ~TpetraCrsGraph() {  }

    //@}

    //! @name Insertion/Removal Methods
    //@{

    //! Insert global indices into the graph.
    void insertGlobalIndices(GlobalOrdinal globalRow, const ArrayView< const GlobalOrdinal > &indices) { }

    //! Insert local indices into the graph.
    void insertLocalIndices(const LocalOrdinal localRow, const ArrayView< const LocalOrdinal > &indices) { }

    //! Remove all graph indices from the specified local row.
    void removeLocalIndices(LocalOrdinal localRow) { }

    //@}

    //! @name Transformational Methods
    //@{

    //! Signal that data entry is complete, specifying domain and range maps.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void fillComplete(const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &domainMap, const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rangeMap, const RCP< ParameterList > &params=null) { }
#else
    void fillComplete(const RCP< const Map<Node > > &domainMap, const RCP< const Map<Node > > &rangeMap, const RCP< ParameterList > &params=null) { }
#endif

    //! Signal that data entry is complete.
    void fillComplete(const RCP< ParameterList > &params=null) { }

    //@}

    //! @name Methods implementing RowGraph.
    //@{

    //! Returns the communicator.
    RCP< const Comm< int > > getComm() const { return Teuchos::null; }

    //! Returns the Map that describes the row distribution in this graph.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  getRowMap() const { return Teuchos::null; }
#else
    RCP< const Map<Node > >  getRowMap() const { return Teuchos::null; }
#endif

    //! Returns the Map that describes the column distribution in this graph.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  getColMap() const { return Teuchos::null; }
#else
    RCP< const Map<Node > >  getColMap() const { return Teuchos::null; }
#endif

    //! Returns the Map associated with the domain of this graph.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  getDomainMap() const { return Teuchos::null; }
#else
    RCP< const Map<Node > >  getDomainMap() const { return Teuchos::null; }
#endif

    //! Returns the Map associated with the domain of this graph.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  getRangeMap() const { return Teuchos::null; }
#else
    RCP< const Map<Node > >  getRangeMap() const { return Teuchos::null; }
#endif

    //! Returns the importer associated with this graph.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Import< LocalOrdinal, GlobalOrdinal, Node > > getImporter() const { return Teuchos::null; }
#else
    RCP< const Import<Node > > getImporter() const { return Teuchos::null; }
#endif

    //! Returns the exporter associated with this graph.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Export< LocalOrdinal, GlobalOrdinal, Node > > getExporter() const { return Teuchos::null; }
#else
    RCP< const Export<Node > > getExporter() const { return Teuchos::null; }
#endif

    //! Returns the number of global rows in the graph.
    global_size_t getGlobalNumRows() const { return 0; }

    //! Returns the number of global columns in the graph.
    global_size_t getGlobalNumCols() const { return 0; }

    //! Returns the number of graph rows owned on the calling node.
    size_t getNodeNumRows() const { return 0; }

    //! Returns the number of columns connected to the locally owned rows of this graph.
    size_t getNodeNumCols() const { return 0; }

    //! Returns the index base for global indices for this graph.
    GlobalOrdinal getIndexBase() const { return 0; }

    //! Returns the global number of entries in the graph.
    global_size_t getGlobalNumEntries() const { return 0; }

    //! Returns the local number of entries in the graph.
    size_t getNodeNumEntries() const { return 0; }

    //! Returns the current number of entries on this node in the specified global row.
    size_t getNumEntriesInGlobalRow(GlobalOrdinal globalRow) const { return 0; }

    //! Returns the current number of entries on this node in the specified local row.
    size_t getNumEntriesInLocalRow(LocalOrdinal localRow) const { return 0; }

    //! Returns the current number of allocated entries for this node in the specified global row .
    size_t getNumAllocatedEntriesInGlobalRow(GlobalOrdinal globalRow) const { return 0; }

    //! Returns the current number of allocated entries on this node in the specified local row.
    size_t getNumAllocatedEntriesInLocalRow(LocalOrdinal localRow) const { return 0; }

    //! Maximum number of entries in all rows over all processes.
    size_t getGlobalMaxNumRowEntries() const { return 0; }

    //! Maximum number of entries in all rows owned by the calling process.
    size_t getNodeMaxNumRowEntries() const { return 0; }

    //! Whether the graph has a column Map.
    bool hasColMap() const { return false; }

    //! Whether column indices are stored using local indices on the calling process.
    bool isLocallyIndexed() const { return false; }

    //! Whether column indices are stored using global indices on the calling process.
    bool isGloballyIndexed() const { return false; }

    //! Whether fillComplete() has been called and the graph is in compute mode.
    bool isFillComplete() const { return false; }

    //! Returns true if storage has been optimized.
    bool isStorageOptimized() const { return false; }

    //! Return a const, nonpersisting view of global indices in the given row.
    void getGlobalRowView(GlobalOrdinal GlobalRow, ArrayView< const GlobalOrdinal > &Indices) const {  }

    //! Return a const, nonpersisting view of local indices in the given row.
    void getLocalRowView(LocalOrdinal LocalRow, ArrayView< const LocalOrdinal > &indices) const {  }

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
    /// \brief Access the local KokkosSparse::StaticCrsGraph data
    local_graph_type getLocalGraph () const {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Xpetra::Exceptions::RuntimeError,
                                 "Epetra does not support Kokkos::StaticCrsGraph!");
      TEUCHOS_UNREACHABLE_RETURN((Kokkos::StaticCrsGraph<LocalOrdinal, Kokkos::LayoutLeft, execution_space>()));
    }
#endif

    //! Dummy implementation for computeGlobalConstants
    void computeGlobalConstants() { }

    //@}

    //! @name Overridden from Teuchos::Describable
    //@{

    //! Return a simple one-line description of this object.
    std::string description() const { return std::string(""); }

    //! Print the object with some verbosity level to an FancyOStream object.
    void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel=Teuchos::Describable::verbLevel_default) const {  }

    //@}

    //! @name Advanced methods, at increased risk of deprecation.
    //@{

    //! Get an ArrayRCP of the row-offsets.
    ArrayRCP< const size_t > getNodeRowPtrs() const { return Teuchos::ArrayRCP< const size_t>(); }

    //@}

    //! Implements DistObject interface
    //{@

    //! Access function for the Tpetra::Map this DistObject was constructed with.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > getMap() const { return Teuchos::null; }
#else
    Teuchos::RCP< const Map<Node > > getMap() const { return Teuchos::null; }
#endif

    //! Import.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doImport(const DistObject<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> &source,
                  const Import< LocalOrdinal, GlobalOrdinal, Node > &importer, CombineMode CM) { }
#else
    void doImport(const DistObject<GlobalOrdinal, Node> &source,
                  const Import<Node > &importer, CombineMode CM) { }
#endif

    //! Export.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doExport(const DistObject<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> &dest,
                  const Import< LocalOrdinal, GlobalOrdinal, Node >& importer, CombineMode CM) { }
#else
    void doExport(const DistObject<GlobalOrdinal, Node> &dest,
                  const Import<Node >& importer, CombineMode CM) { }
#endif

    //! Import (using an Exporter).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doImport(const DistObject<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> &source,
                  const Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM) { }
#else
    void doImport(const DistObject<GlobalOrdinal, Node> &source,
                  const Export<Node >& exporter, CombineMode CM) { }
#endif

    //! Export (using an Importer).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doExport(const DistObject<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> &dest,
                  const Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM) { }
#else
    void doExport(const DistObject<GlobalOrdinal, Node> &dest,
                  const Export<Node >& exporter, CombineMode CM) { }
#endif

    // @}

    //! @name Xpetra specific
    //@{

    //! TpetraCrsGraph constructor to wrap a Tpetra::CrsGraph object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraCrsGraph(const Teuchos::RCP<Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node> > &graph)  {
#else
    TpetraCrsGraph(const Teuchos::RCP<Tpetra::CrsGraph<Node> > &graph)  {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraCrsGraph<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

    //! Get the underlying Tpetra graph
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node> > getTpetra_CrsGraph() const { return Teuchos::null; }
#else
    RCP< const Tpetra::CrsGraph<Node> > getTpetra_CrsGraph() const { return Teuchos::null; }
#endif

    //@}
  }; // TpetraCrsGraph class (specialization for GO=long long and NO=EpetraNode)
#endif

#endif // HAVE_XPETRA_EPETRA


} // Xpetra namespace
#endif //XPETRA_TPETRACRSGRAPH_DEF_HPP

