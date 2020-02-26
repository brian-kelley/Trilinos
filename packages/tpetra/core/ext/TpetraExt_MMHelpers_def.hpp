// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
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
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER

#ifndef TPETRA_MMHELPERS_DEF_HPP
#define TPETRA_MMHELPERS_DEF_HPP

#include "TpetraExt_MMHelpers_decl.hpp"
#include "Teuchos_VerboseObject.hpp"

/*! \file TpetraExt_MMHelpers_def.hpp

    The implementations for the MatrixMatrix helper classes.
 */
namespace Tpetra {

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Node>::CrsMatrixStruct()
#else
template <class Scalar, class Node>
CrsMatrixStruct<Scalar, Node>::CrsMatrixStruct()
#endif
{
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Node>::~CrsMatrixStruct()
#else
template <class Scalar, class Node>
CrsMatrixStruct<Scalar, Node>::~CrsMatrixStruct()
#endif
{
  deleteContents();
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
template <class Scalar, class Node>
void CrsMatrixStruct<Scalar, Node>::
#endif
deleteContents ()
{
  importMatrix.reset();
  origMatrix = Teuchos::null;
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
int dumpCrsMatrixStruct (const CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Node>& M)
#else
template <class Scalar, class Node>
int dumpCrsMatrixStruct (const CrsMatrixStruct<Scalar, Node>& M)
#endif
{
  std::cout << "proc " << M.rowMap->Comm().MyPID()<<std::endl;
  std::cout << "numRows: " << M.numRows<<std::endl;
  for(LocalOrdinal i=0; i<M.numRows; ++i) {
    for(LocalOrdinal j=0; j<M.numEntriesPerRow[i]; ++j) {
      std::cout << "   "<<M.rowMap->GID(i)<<"   "
                <<M.colMap->GID(M.indices[i][j])<<"   "<<M.values[i][j]<<std::endl;
    }
  }

  return 0;
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
CrsWrapper_CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
CrsWrapper_CrsMatrix (CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>& crsmatrix)
#else
template<class Scalar, class Node>
CrsWrapper_CrsMatrix<Scalar, Node>::
CrsWrapper_CrsMatrix (CrsMatrix<Scalar, Node>& crsmatrix)
#endif
 : crsmat_ (crsmatrix)
{
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
CrsWrapper_CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::~CrsWrapper_CrsMatrix()
#else
template<class Scalar, class Node>
CrsWrapper_CrsMatrix<Scalar, Node>::~CrsWrapper_CrsMatrix()
#endif
{
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> >
CrsWrapper_CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getRowMap() const
#else
template<class Scalar, class Node>
Teuchos::RCP<const Map<Node> >
CrsWrapper_CrsMatrix<Scalar, Node>::getRowMap() const
#endif
{
  return crsmat_.getRowMap();
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool CrsWrapper_CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Scalar, class Node>
bool CrsWrapper_CrsMatrix<Scalar, Node>::
#endif
isFillComplete ()
{
  return crsmat_.isFillComplete ();
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
CrsWrapper_CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
CrsWrapper_CrsMatrix<Scalar, Node>::
#endif
insertGlobalValues (GlobalOrdinal globalRow,
                    const Teuchos::ArrayView<const GlobalOrdinal> &indices,
                    const Teuchos::ArrayView<const Scalar> &values)
{
  crsmat_.insertGlobalValues (globalRow, indices, values);
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
CrsWrapper_CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
CrsWrapper_CrsMatrix<Scalar, Node>::
#endif
sumIntoGlobalValues (GlobalOrdinal globalRow,
                     const Teuchos::ArrayView<const GlobalOrdinal> &indices,
                     const Teuchos::ArrayView<const Scalar> &values)
{
  crsmat_.sumIntoGlobalValues (globalRow, indices, values);
}



#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
CrsWrapper_GraphBuilder<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
CrsWrapper_GraphBuilder (const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> >& map)
#else
template<class Scalar, class Node>
CrsWrapper_GraphBuilder<Scalar, Node>::
CrsWrapper_GraphBuilder (const Teuchos::RCP<const Map<Node> >& map)
#endif
 : graph_(),
   rowmap_(map),
   max_row_length_(0)
{
  Teuchos::ArrayView<const GlobalOrdinal> rows = map->getNodeElementList ();
  const LocalOrdinal numRows = static_cast<LocalOrdinal> (rows.size ());
  for (LocalOrdinal i = 0; i < numRows; ++i) {
    graph_[rows[i]] = new std::set<GlobalOrdinal>;
  }
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
CrsWrapper_GraphBuilder<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Scalar, class Node>
CrsWrapper_GraphBuilder<Scalar, Node>::
#endif
~CrsWrapper_GraphBuilder ()
{
  typename std::map<GlobalOrdinal,std::set<GlobalOrdinal>*>::iterator
    iter = graph_.begin(), iter_end = graph_.end();
  for (; iter != iter_end; ++iter) {
    delete iter->second;
  }
  graph_.clear ();
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool CrsWrapper_GraphBuilder<Scalar, LocalOrdinal, GlobalOrdinal, Node>::isFillComplete()
#else
template<class Scalar, class Node>
bool CrsWrapper_GraphBuilder<Scalar, Node>::isFillComplete()
#endif
{
  return false;
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
CrsWrapper_GraphBuilder<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
CrsWrapper_GraphBuilder<Scalar, Node>::
#endif
insertGlobalValues (GlobalOrdinal globalRow,
                    const Teuchos::ArrayView<const GlobalOrdinal> &indices,
                    const Teuchos::ArrayView<const Scalar> &/* values */)
{
  typename std::map<GlobalOrdinal,std::set<GlobalOrdinal>*>::iterator
    iter = graph_.find (globalRow);

  TEUCHOS_TEST_FOR_EXCEPTION(
    iter == graph_.end(), std::runtime_error,
    "Tpetra::CrsWrapper_GraphBuilder::insertGlobalValues could not find row "
    << globalRow << " in the graph. Super bummer man. Hope you figure it out.");

  std::set<GlobalOrdinal>& cols = * (iter->second);

  for (typename Teuchos::ArrayView<const GlobalOrdinal>::size_type i = 0;
       i < indices.size (); ++i) {
    cols.insert (indices[i]);
  }

  const global_size_t row_length = static_cast<global_size_t> (cols.size ());
  if (row_length > max_row_length_) {
    max_row_length_ = row_length;
  }
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
CrsWrapper_GraphBuilder<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
CrsWrapper_GraphBuilder<Scalar, Node>::
#endif
sumIntoGlobalValues (GlobalOrdinal globalRow,
                     const Teuchos::ArrayView<const GlobalOrdinal> &indices,
                     const Teuchos::ArrayView<const Scalar> &values)
{
  insertGlobalValues (globalRow, indices, values);
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
std::map<GlobalOrdinal,std::set<GlobalOrdinal>*>&
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
CrsWrapper_GraphBuilder<Scalar, LocalOrdinal, GlobalOrdinal, Node>::get_graph ()
#else
CrsWrapper_GraphBuilder<Scalar, Node>::get_graph ()
#endif
{
  return graph_;
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
insert_matrix_locations (CrsWrapper_GraphBuilder<Scalar, LocalOrdinal, GlobalOrdinal, Node>& graphbuilder,
                         CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>& C)
#else
insert_matrix_locations (CrsWrapper_GraphBuilder<Scalar, Node>& graphbuilder,
                         CrsMatrix<Scalar, Node>& C)
#endif
{
  global_size_t max_row_length = graphbuilder.get_max_row_length();
  if (max_row_length < 1) return;

  Teuchos::Array<GlobalOrdinal> indices(max_row_length);
  Teuchos::Array<Scalar> zeros(max_row_length, Teuchos::ScalarTraits<Scalar>::zero());

  typedef std::map<GlobalOrdinal,std::set<GlobalOrdinal>*> Graph;
  typedef typename Graph::iterator GraphIter;
  Graph& graph = graphbuilder.get_graph ();

  const GraphIter iter_end = graph.end ();
  for (GraphIter iter = graph.begin (); iter != iter_end; ++iter) {
    const GlobalOrdinal row = iter->first;
    const std::set<GlobalOrdinal>& cols = * (iter->second);
    // "copy" entries out of set into contiguous array storage
    const size_t num_entries = std::copy (cols.begin (), cols.end (), indices.begin ()) - indices.begin ();
    // insert zeros into the result matrix at the appropriate locations
    C.insertGlobalValues (row, indices (0, num_entries), zeros (0, num_entries));
  }
}

} // namespace Tpetra

//
// Explicit instantiation macro
//
// Must be expanded from within the Tpetra namespace!
//

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
#define TPETRA_CRSMATRIXSTRUCT_INSTANT(SCALAR,LO,GO,NODE) \
#else
#define TPETRA_CRSMATRIXSTRUCT_INSTANT(SCALAR,NODE) \
#endif
  \
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template class CrsMatrixStruct< SCALAR , LO , GO , NODE >;
#else
  template class CrsMatrixStruct< SCALAR , NODE >;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
#define TPETRA_CRSWRAPPER_INSTANT(SCALAR,LO,GO,NODE) \
#else
#define TPETRA_CRSWRAPPER_INSTANT(SCALAR,NODE) \
#endif
  \
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template class CrsWrapper< SCALAR , LO , GO , NODE >;
#else
  template class CrsWrapper< SCALAR , NODE >;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
#define TPETRA_CRSWRAPPER_CRSMATRIX_INSTANT(SCALAR,LO,GO,NODE) \
#else
#define TPETRA_CRSWRAPPER_CRSMATRIX_INSTANT(SCALAR,NODE) \
#endif
  \
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template class CrsWrapper_CrsMatrix< SCALAR , LO , GO , NODE >;
#else
  template class CrsWrapper_CrsMatrix< SCALAR , NODE >;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
#define TPETRA_CRSWRAPPER_GRAPHBUILDER_INSTANT(SCALAR,LO,GO,NODE) \
#else
#define TPETRA_CRSWRAPPER_GRAPHBUILDER_INSTANT(SCALAR,NODE) \
#endif
  \
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template class CrsWrapper_GraphBuilder< SCALAR , LO , GO , NODE >;
#else
  template class CrsWrapper_GraphBuilder< SCALAR , NODE >;
#endif

#endif // TPETRA_MMHELPERS_DEF_HPP
