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
#ifndef XPETRA_TPETRACRSMATRIX_DEF_HPP
#define XPETRA_TPETRACRSMATRIX_DEF_HPP

#include "Xpetra_TpetraCrsMatrix_decl.hpp"
#include "Tpetra_Details_residual.hpp"

namespace Xpetra {

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsMatrix(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &params)
      : mtx_ (Teuchos::rcp (new Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> (toTpetra(rowMap), maxNumEntriesPerRow, Tpetra::StaticProfile, params))) {  }
#else
    template <class Scalar, class Node>
    TpetraCrsMatrix<Scalar,Node>::TpetraCrsMatrix(const Teuchos::RCP< const Map<Node > > &rowMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &params)
      : mtx_ (Teuchos::rcp (new Tpetra::CrsMatrix<Scalar, Node> (toTpetra(rowMap), maxNumEntriesPerRow, Tpetra::StaticProfile, params))) {  }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsMatrix(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &params)
      : mtx_(Teuchos::rcp(new Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> (toTpetra(rowMap), NumEntriesPerRowToAlloc(), Tpetra::StaticProfile, params))) {  }
#else
    template <class Scalar, class Node>
    TpetraCrsMatrix<Scalar,Node>::TpetraCrsMatrix(const Teuchos::RCP< const Map<Node > > &rowMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &params)
      : mtx_(Teuchos::rcp(new Tpetra::CrsMatrix<Scalar, Node> (toTpetra(rowMap), NumEntriesPerRowToAlloc(), Tpetra::StaticProfile, params))) {  }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsMatrix(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &params)
      : mtx_(Teuchos::rcp(new Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(toTpetra(rowMap), toTpetra(colMap), maxNumEntriesPerRow, Tpetra::StaticProfile, params))) {  }
#else
    template <class Scalar, class Node>
    TpetraCrsMatrix<Scalar,Node>::TpetraCrsMatrix(const Teuchos::RCP< const Map<Node > > &rowMap, const Teuchos::RCP< const Map<Node > > &colMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &params)
      : mtx_(Teuchos::rcp(new Tpetra::CrsMatrix<Scalar, Node>(toTpetra(rowMap), toTpetra(colMap), maxNumEntriesPerRow, Tpetra::StaticProfile, params))) {  }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsMatrix(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &params)
      : mtx_(Teuchos::rcp(new Tpetra::CrsMatrix< Scalar, LocalOrdinal, GlobalOrdinal, Node >(toTpetra(rowMap), toTpetra(colMap), NumEntriesPerRowToAlloc(), Tpetra::StaticProfile, params))) {  }
#else
    template <class Scalar, class Node>
    TpetraCrsMatrix<Scalar,Node>::TpetraCrsMatrix(const Teuchos::RCP< const Map<Node > > &rowMap, const Teuchos::RCP< const Map<Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &params)
      : mtx_(Teuchos::rcp(new Tpetra::CrsMatrix< Scalar, Node >(toTpetra(rowMap), toTpetra(colMap), NumEntriesPerRowToAlloc(), Tpetra::StaticProfile, params))) {  }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsMatrix(const Teuchos::RCP< const CrsGraph< LocalOrdinal, GlobalOrdinal, Node > > &graph, const Teuchos::RCP< Teuchos::ParameterList > &params)
      : mtx_(Teuchos::rcp(new Tpetra::CrsMatrix< Scalar, LocalOrdinal, GlobalOrdinal, Node >(toTpetra(graph), params))) {  }
#else
    template <class Scalar, class Node>
    TpetraCrsMatrix<Scalar,Node>::TpetraCrsMatrix(const Teuchos::RCP< const CrsGraph<Node > > &graph, const Teuchos::RCP< Teuchos::ParameterList > &params)
      : mtx_(Teuchos::rcp(new Tpetra::CrsMatrix< Scalar, Node >(toTpetra(graph), params))) {  }
#endif



#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsMatrix(const Teuchos::RCP<const CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& sourceMatrix,
                    const Import<LocalOrdinal,GlobalOrdinal,Node> & importer,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap,
#else
    template <class Scalar, class Node>
    TpetraCrsMatrix<Scalar,Node>::TpetraCrsMatrix(const Teuchos::RCP<const CrsMatrix<Scalar,Node> >& sourceMatrix,
                    const Import<Node> & importer,
                    const Teuchos::RCP<const Map<Node> >& domainMap,
                    const Teuchos::RCP<const Map<Node> >& rangeMap,
#endif
                    const Teuchos::RCP<Teuchos::ParameterList>& params)
    {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      typedef Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> MyTpetraCrsMatrix;
#else
      typedef Tpetra::CrsMatrix<Scalar,Node> MyTpetraCrsMatrix;
#endif
      XPETRA_DYNAMIC_CAST(const TpetraCrsMatrixClass, *sourceMatrix, tSourceMatrix, "Xpetra::TpetraCrsMatrix constructor only accepts Xpetra::TpetraCrsMatrix as the input argument.");//TODO: remove and use toTpetra()
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP< const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > v = tSourceMatrix.getTpetra_CrsMatrix();
#else
      RCP< const Tpetra::CrsMatrix<Scalar, Node> > v = tSourceMatrix.getTpetra_CrsMatrix();
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > myDomainMap = domainMap!=Teuchos::null ? toTpetra(domainMap) : Teuchos::null;
      RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > myRangeMap  = rangeMap!=Teuchos::null  ? toTpetra(rangeMap)  : Teuchos::null;
#else
      RCP<const Tpetra::Map<Node> > myDomainMap = domainMap!=Teuchos::null ? toTpetra(domainMap) : Teuchos::null;
      RCP<const Tpetra::Map<Node> > myRangeMap  = rangeMap!=Teuchos::null  ? toTpetra(rangeMap)  : Teuchos::null;
#endif
      mtx_=Tpetra::importAndFillCompleteCrsMatrix<MyTpetraCrsMatrix>(tSourceMatrix.getTpetra_CrsMatrix(),toTpetra(importer),myDomainMap,myRangeMap,params);
      bool restrictComm=false;
      if(!params.is_null()) restrictComm = params->get("Restrict Communicator",restrictComm);
      if(restrictComm && mtx_->getRowMap().is_null()) mtx_=Teuchos::null;

    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsMatrix(const Teuchos::RCP<const CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& sourceMatrix,
                    const Export<LocalOrdinal,GlobalOrdinal,Node> & exporter,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap,
#else
    template <class Scalar, class Node>
    TpetraCrsMatrix<Scalar,Node>::TpetraCrsMatrix(const Teuchos::RCP<const CrsMatrix<Scalar,Node> >& sourceMatrix,
                    const Export<Node> & exporter,
                    const Teuchos::RCP<const Map<Node> >& domainMap,
                    const Teuchos::RCP<const Map<Node> >& rangeMap,
#endif
                    const Teuchos::RCP<Teuchos::ParameterList>& params)
    {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      typedef Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> MyTpetraCrsMatrix;
#else
      typedef Tpetra::CrsMatrix<Scalar,Node> MyTpetraCrsMatrix;
#endif
      XPETRA_DYNAMIC_CAST(const TpetraCrsMatrixClass, *sourceMatrix, tSourceMatrix, "Xpetra::TpetraCrsMatrix constructor only accepts Xpetra::TpetraCrsMatrix as the input argument.");//TODO: remove and use toTpetra()
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP< const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > v = tSourceMatrix.getTpetra_CrsMatrix();
#else
      RCP< const Tpetra::CrsMatrix<Scalar, Node> > v = tSourceMatrix.getTpetra_CrsMatrix();
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > myDomainMap = domainMap!=Teuchos::null ? toTpetra(domainMap) : Teuchos::null;
      RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > myRangeMap  = rangeMap!=Teuchos::null  ? toTpetra(rangeMap)  : Teuchos::null;
#else
      RCP<const Tpetra::Map<Node> > myDomainMap = domainMap!=Teuchos::null ? toTpetra(domainMap) : Teuchos::null;
      RCP<const Tpetra::Map<Node> > myRangeMap  = rangeMap!=Teuchos::null  ? toTpetra(rangeMap)  : Teuchos::null;
#endif
      mtx_=Tpetra::exportAndFillCompleteCrsMatrix<MyTpetraCrsMatrix>(tSourceMatrix.getTpetra_CrsMatrix(),toTpetra(exporter),myDomainMap,myRangeMap,params);

    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsMatrix(const Teuchos::RCP<const CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& sourceMatrix,
                    const Import<LocalOrdinal,GlobalOrdinal,Node> & RowImporter,
                    const Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Node> > DomainImporter,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap,
#else
    template <class Scalar, class Node>
    TpetraCrsMatrix<Scalar,Node>::TpetraCrsMatrix(const Teuchos::RCP<const CrsMatrix<Scalar,Node> >& sourceMatrix,
                    const Import<Node> & RowImporter,
                    const Teuchos::RCP<const Import<Node> > DomainImporter,
                    const Teuchos::RCP<const Map<Node> >& domainMap,
                    const Teuchos::RCP<const Map<Node> >& rangeMap,
#endif
                    const Teuchos::RCP<Teuchos::ParameterList>& params)
    {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      typedef Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> MyTpetraCrsMatrix;
#else
      typedef Tpetra::CrsMatrix<Scalar,Node> MyTpetraCrsMatrix;
#endif
      XPETRA_DYNAMIC_CAST(const TpetraCrsMatrixClass, *sourceMatrix, tSourceMatrix, "Xpetra::TpetraCrsMatrix constructor only accepts Xpetra::TpetraCrsMatrix as the input argument.");//TODO: remove and use toTpetra()
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP< const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > v = tSourceMatrix.getTpetra_CrsMatrix();
#else
      RCP< const Tpetra::CrsMatrix<Scalar, Node> > v = tSourceMatrix.getTpetra_CrsMatrix();
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > myDomainMap = domainMap!=Teuchos::null ? toTpetra(domainMap) : Teuchos::null;
      RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > myRangeMap  = rangeMap!=Teuchos::null  ? toTpetra(rangeMap)  : Teuchos::null;
#else
      RCP<const Tpetra::Map<Node> > myDomainMap = domainMap!=Teuchos::null ? toTpetra(domainMap) : Teuchos::null;
      RCP<const Tpetra::Map<Node> > myRangeMap  = rangeMap!=Teuchos::null  ? toTpetra(rangeMap)  : Teuchos::null;
#endif

      mtx_=Tpetra::importAndFillCompleteCrsMatrix<MyTpetraCrsMatrix>(tSourceMatrix.getTpetra_CrsMatrix(),toTpetra(RowImporter),toTpetra(*DomainImporter),myDomainMap,myRangeMap,params);
      bool restrictComm=false;
      if(!params.is_null()) restrictComm = params->get("Restrict Communicator",restrictComm);
      if(restrictComm && mtx_->getRowMap().is_null()) mtx_=Teuchos::null;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsMatrix(const Teuchos::RCP<const CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& sourceMatrix,
                    const Export<LocalOrdinal,GlobalOrdinal,Node> & RowExporter,
                    const Teuchos::RCP<const Export<LocalOrdinal,GlobalOrdinal,Node> > DomainExporter,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap,
#else
    template <class Scalar, class Node>
    TpetraCrsMatrix<Scalar,Node>::TpetraCrsMatrix(const Teuchos::RCP<const CrsMatrix<Scalar,Node> >& sourceMatrix,
                    const Export<Node> & RowExporter,
                    const Teuchos::RCP<const Export<Node> > DomainExporter,
                    const Teuchos::RCP<const Map<Node> >& domainMap,
                    const Teuchos::RCP<const Map<Node> >& rangeMap,
#endif
                    const Teuchos::RCP<Teuchos::ParameterList>& params)
    {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      typedef Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> MyTpetraCrsMatrix;
#else
      typedef Tpetra::CrsMatrix<Scalar,Node> MyTpetraCrsMatrix;
#endif
      XPETRA_DYNAMIC_CAST(const TpetraCrsMatrixClass, *sourceMatrix, tSourceMatrix, "Xpetra::TpetraCrsMatrix constructor only accepts Xpetra::TpetraCrsMatrix as the input argument.");//TODO: remove and use toTpetra()
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP< const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > v = tSourceMatrix.getTpetra_CrsMatrix();
#else
      RCP< const Tpetra::CrsMatrix<Scalar, Node> > v = tSourceMatrix.getTpetra_CrsMatrix();
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > myDomainMap = domainMap!=Teuchos::null ? toTpetra(domainMap) : Teuchos::null;
      RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > myRangeMap  = rangeMap!=Teuchos::null  ? toTpetra(rangeMap)  : Teuchos::null;
#else
      RCP<const Tpetra::Map<Node> > myDomainMap = domainMap!=Teuchos::null ? toTpetra(domainMap) : Teuchos::null;
      RCP<const Tpetra::Map<Node> > myRangeMap  = rangeMap!=Teuchos::null  ? toTpetra(rangeMap)  : Teuchos::null;
#endif

      mtx_=Tpetra::exportAndFillCompleteCrsMatrix<MyTpetraCrsMatrix>(tSourceMatrix.getTpetra_CrsMatrix(),toTpetra(RowExporter),toTpetra(*DomainExporter),myDomainMap,myRangeMap,params);
    }
 
///////////////////////////////////////////////////////////////////////////////////////


#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef HAVE_XPETRA_TPETRA
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsMatrix (const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rowMap,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& colMap,
#else
    template <class Scalar, class Node>
    TpetraCrsMatrix<Scalar,Node>::TpetraCrsMatrix (const Teuchos::RCP<const Map<Node> >& rowMap,
        const Teuchos::RCP<const Map<Node> >& colMap,
#endif
        const local_matrix_type& lclMatrix,
        const Teuchos::RCP<Teuchos::ParameterList>& params)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    : mtx_(Teuchos::rcp(new Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(toTpetra(rowMap), toTpetra(colMap), lclMatrix, params))) {  }
#else
    : mtx_(Teuchos::rcp(new Tpetra::CrsMatrix<Scalar, Node>(toTpetra(rowMap), toTpetra(colMap), lclMatrix, params))) {  }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsMatrix (
#else
    template <class Scalar, class Node>
    TpetraCrsMatrix<Scalar,Node>::TpetraCrsMatrix (
#endif
        const local_matrix_type& lclMatrix,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rowMap,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& colMap,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap,
#else
        const Teuchos::RCP<const Map<Node> >& rowMap,
        const Teuchos::RCP<const Map<Node> >& colMap,
        const Teuchos::RCP<const Map<Node> >& domainMap,
        const Teuchos::RCP<const Map<Node> >& rangeMap,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    : mtx_(Teuchos::rcp(new Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(lclMatrix, toTpetra(rowMap), toTpetra(colMap), toTpetra(domainMap), toTpetra(rangeMap), params))) {  }
#else
    : mtx_(Teuchos::rcp(new Tpetra::CrsMatrix<Scalar, Node>(lclMatrix, toTpetra(rowMap), toTpetra(colMap), toTpetra(domainMap), toTpetra(rangeMap), params))) {  }
#endif
#endif
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::~TpetraCrsMatrix() {  }
#else
    template <class Scalar, class Node>
    TpetraCrsMatrix<Scalar,Node>::~TpetraCrsMatrix() {  }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::insertGlobalValues(GlobalOrdinal globalRow, const ArrayView< const GlobalOrdinal > &cols, const ArrayView< const Scalar > &vals) { XPETRA_MONITOR("TpetraCrsMatrix::insertGlobalValues"); mtx_->insertGlobalValues(globalRow, cols, vals); }
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::insertGlobalValues(GlobalOrdinal globalRow, const ArrayView< const GlobalOrdinal > &cols, const ArrayView< const Scalar > &vals) { XPETRA_MONITOR("TpetraCrsMatrix::insertGlobalValues"); mtx_->insertGlobalValues(globalRow, cols, vals); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::insertLocalValues(LocalOrdinal localRow, const ArrayView< const LocalOrdinal > &cols, const ArrayView< const Scalar > &vals) { XPETRA_MONITOR("TpetraCrsMatrix::insertLocalValues"); mtx_->insertLocalValues(localRow, cols, vals); }
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::insertLocalValues(LocalOrdinal localRow, const ArrayView< const LocalOrdinal > &cols, const ArrayView< const Scalar > &vals) { XPETRA_MONITOR("TpetraCrsMatrix::insertLocalValues"); mtx_->insertLocalValues(localRow, cols, vals); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::replaceGlobalValues(GlobalOrdinal globalRow, const ArrayView< const GlobalOrdinal > &cols, const ArrayView< const Scalar > &vals) { XPETRA_MONITOR("TpetraCrsMatrix::replaceGlobalValues"); mtx_->replaceGlobalValues(globalRow, cols, vals); }
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::replaceGlobalValues(GlobalOrdinal globalRow, const ArrayView< const GlobalOrdinal > &cols, const ArrayView< const Scalar > &vals) { XPETRA_MONITOR("TpetraCrsMatrix::replaceGlobalValues"); mtx_->replaceGlobalValues(globalRow, cols, vals); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::replaceLocalValues (LocalOrdinal localRow,
#else
    TpetraCrsMatrix<Scalar,Node>::replaceLocalValues (LocalOrdinal localRow,
#endif
                        const ArrayView<const LocalOrdinal> &cols,
                        const ArrayView<const Scalar> &vals)
    {
      XPETRA_MONITOR("TpetraCrsMatrix::replaceLocalValues");
      typedef typename ArrayView<const LocalOrdinal>::size_type size_type;
      const LocalOrdinal numValid =
        mtx_->replaceLocalValues (localRow, cols, vals);
      TEUCHOS_TEST_FOR_EXCEPTION(
        static_cast<size_type> (numValid) != cols.size (), std::runtime_error,
        "replaceLocalValues returned " << numValid << " != cols.size() = " <<
        cols.size () << ".");
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::setAllToScalar(const Scalar &alpha) { XPETRA_MONITOR("TpetraCrsMatrix::setAllToScalar"); mtx_->setAllToScalar(alpha); }
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::setAllToScalar(const Scalar &alpha) { XPETRA_MONITOR("TpetraCrsMatrix::setAllToScalar"); mtx_->setAllToScalar(alpha); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::scale(const Scalar &alpha) { XPETRA_MONITOR("TpetraCrsMatrix::scale"); mtx_->scale(alpha); }
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::scale(const Scalar &alpha) { XPETRA_MONITOR("TpetraCrsMatrix::scale"); mtx_->scale(alpha); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::allocateAllValues(size_t numNonZeros,ArrayRCP<size_t> & rowptr, ArrayRCP<LocalOrdinal> & colind, ArrayRCP<Scalar> & values)
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::allocateAllValues(size_t numNonZeros,ArrayRCP<size_t> & rowptr, ArrayRCP<LocalOrdinal> & colind, ArrayRCP<Scalar> & values)
#endif
    { XPETRA_MONITOR("TpetraCrsMatrix::allocateAllValues"); rowptr.resize(getNodeNumRows()+1); colind.resize(numNonZeros); values.resize(numNonZeros);}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::setAllValues(const ArrayRCP<size_t> & rowptr, const ArrayRCP<LocalOrdinal> & colind, const ArrayRCP<Scalar> & values)
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::setAllValues(const ArrayRCP<size_t> & rowptr, const ArrayRCP<LocalOrdinal> & colind, const ArrayRCP<Scalar> & values)
#endif
    { XPETRA_MONITOR("TpetraCrsMatrix::setAllValues"); mtx_->setAllValues(rowptr,colind,values); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getAllValues(ArrayRCP<const size_t>& rowptr, ArrayRCP<const LocalOrdinal>& colind, ArrayRCP<const Scalar>& values) const
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::getAllValues(ArrayRCP<const size_t>& rowptr, ArrayRCP<const LocalOrdinal>& colind, ArrayRCP<const Scalar>& values) const
#endif
    { XPETRA_MONITOR("TpetraCrsMatrix::getAllValues"); mtx_->getAllValues(rowptr,colind,values); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    bool TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::haveGlobalConstants() const
#else
    template <class Scalar, class Node>
    bool TpetraCrsMatrix<Scalar,Node>::haveGlobalConstants() const
#endif
    { return mtx_->haveGlobalConstants();}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::resumeFill(const RCP< ParameterList > &params) { XPETRA_MONITOR("TpetraCrsMatrix::resumeFill"); mtx_->resumeFill(params); }
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::resumeFill(const RCP< ParameterList > &params) { XPETRA_MONITOR("TpetraCrsMatrix::resumeFill"); mtx_->resumeFill(params); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::fillComplete(const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &domainMap, const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rangeMap, const RCP< ParameterList > &params) { XPETRA_MONITOR("TpetraCrsMatrix::fillComplete"); mtx_->fillComplete(toTpetra(domainMap), toTpetra(rangeMap), params); }
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::fillComplete(const RCP< const Map<Node > > &domainMap, const RCP< const Map<Node > > &rangeMap, const RCP< ParameterList > &params) { XPETRA_MONITOR("TpetraCrsMatrix::fillComplete"); mtx_->fillComplete(toTpetra(domainMap), toTpetra(rangeMap), params); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::fillComplete(const RCP< ParameterList > &params) { XPETRA_MONITOR("TpetraCrsMatrix::fillComplete"); mtx_->fillComplete(params); }
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::fillComplete(const RCP< ParameterList > &params) { XPETRA_MONITOR("TpetraCrsMatrix::fillComplete"); mtx_->fillComplete(params); }
#endif


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::replaceDomainMapAndImporter(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >& newDomainMap, Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Node> >  & newImporter) {
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::replaceDomainMapAndImporter(const Teuchos::RCP< const Map<Node > >& newDomainMap, Teuchos::RCP<const Import<Node> >  & newImporter) {
#endif
      XPETRA_MONITOR("TpetraCrsMatrix::replaceDomainMapAndImporter");
      XPETRA_DYNAMIC_CAST( const TpetraImportClass , *newImporter, tImporter, "Xpetra::TpetraCrsMatrix::replaceDomainMapAndImporter only accepts Xpetra::TpetraImport.");
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<const Tpetra::Import<LocalOrdinal,GlobalOrdinal,Node> > myImport = tImporter.getTpetra_Import();
#else
      RCP<const Tpetra::Import<Node> > myImport = tImporter.getTpetra_Import();
#endif
            mtx_->replaceDomainMapAndImporter( toTpetra(newDomainMap),myImport);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::expertStaticFillComplete(const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & domainMap,
                                  const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & rangeMap,
                                  const RCP<const Import<LocalOrdinal,GlobalOrdinal,Node> > &importer,
                                  const RCP<const Export<LocalOrdinal,GlobalOrdinal,Node> > &exporter,
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::expertStaticFillComplete(const RCP<const Map<Node> > & domainMap,
                                  const RCP<const Map<Node> > & rangeMap,
                                  const RCP<const Import<Node> > &importer,
                                  const RCP<const Export<Node> > &exporter,
#endif
                                  const RCP<ParameterList> &params) {
      XPETRA_MONITOR("TpetraCrsMatrix::expertStaticFillComplete");
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<const Tpetra::Import<LocalOrdinal,GlobalOrdinal,Node> > myImport;
      RCP<const Tpetra::Export<LocalOrdinal,GlobalOrdinal,Node> > myExport;
#else
      RCP<const Tpetra::Import<Node> > myImport;
      RCP<const Tpetra::Export<Node> > myExport;
#endif

      if(importer!=Teuchos::null) {
        XPETRA_DYNAMIC_CAST( const TpetraImportClass , *importer, tImporter, "Xpetra::TpetraCrsMatrix::expertStaticFillComplete only accepts Xpetra::TpetraImport.");
        myImport = tImporter.getTpetra_Import();
      }
      if(exporter!=Teuchos::null) {
        XPETRA_DYNAMIC_CAST( const TpetraExportClass , *exporter, tExporter, "Xpetra::TpetraCrsMatrix::expertStaticFillComplete only accepts Xpetra::TpetraExport.");
        myExport = tExporter.getTpetra_Export();
      }

      mtx_->expertStaticFillComplete(toTpetra(domainMap),toTpetra(rangeMap),myImport,myExport,params);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getRowMap() const { XPETRA_MONITOR("TpetraCrsMatrix::getRowMap"); return toXpetra(mtx_->getRowMap()); }
#else
    template <class Scalar, class Node>
    const RCP< const Map<Node > >  TpetraCrsMatrix<Scalar,Node>::getRowMap() const { XPETRA_MONITOR("TpetraCrsMatrix::getRowMap"); return toXpetra(mtx_->getRowMap()); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getColMap() const { XPETRA_MONITOR("TpetraCrsMatrix::getColMap"); return toXpetra(mtx_->getColMap()); }
#else
    template <class Scalar, class Node>
    const RCP< const Map<Node > >  TpetraCrsMatrix<Scalar,Node>::getColMap() const { XPETRA_MONITOR("TpetraCrsMatrix::getColMap"); return toXpetra(mtx_->getColMap()); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    RCP< const CrsGraph< LocalOrdinal, GlobalOrdinal, Node > > TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getCrsGraph() const { XPETRA_MONITOR("TpetraCrsMatrix::getCrsGraph"); return toXpetra(mtx_->getCrsGraph()); }
#else
    template <class Scalar, class Node>
    RCP< const CrsGraph<Node > > TpetraCrsMatrix<Scalar,Node>::getCrsGraph() const { XPETRA_MONITOR("TpetraCrsMatrix::getCrsGraph"); return toXpetra(mtx_->getCrsGraph()); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    global_size_t TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getGlobalNumRows() const { XPETRA_MONITOR("TpetraCrsMatrix::getGlobalNumRows"); return mtx_->getGlobalNumRows(); }
#else
    template <class Scalar, class Node>
    global_size_t TpetraCrsMatrix<Scalar,Node>::getGlobalNumRows() const { XPETRA_MONITOR("TpetraCrsMatrix::getGlobalNumRows"); return mtx_->getGlobalNumRows(); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    global_size_t TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getGlobalNumCols() const { XPETRA_MONITOR("TpetraCrsMatrix::getGlobalNumCols"); return mtx_->getGlobalNumCols(); }
#else
    template <class Scalar, class Node>
    global_size_t TpetraCrsMatrix<Scalar,Node>::getGlobalNumCols() const { XPETRA_MONITOR("TpetraCrsMatrix::getGlobalNumCols"); return mtx_->getGlobalNumCols(); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    size_t TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getNodeNumRows() const { XPETRA_MONITOR("TpetraCrsMatrix::getNodeNumRows"); return mtx_->getNodeNumRows(); }
#else
    template <class Scalar, class Node>
    size_t TpetraCrsMatrix<Scalar,Node>::getNodeNumRows() const { XPETRA_MONITOR("TpetraCrsMatrix::getNodeNumRows"); return mtx_->getNodeNumRows(); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    size_t TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getNodeNumCols() const { XPETRA_MONITOR("TpetraCrsMatrix::getNodeNumCols"); return mtx_->getNodeNumCols(); }
#else
    template <class Scalar, class Node>
    size_t TpetraCrsMatrix<Scalar,Node>::getNodeNumCols() const { XPETRA_MONITOR("TpetraCrsMatrix::getNodeNumCols"); return mtx_->getNodeNumCols(); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    global_size_t TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getGlobalNumEntries() const { XPETRA_MONITOR("TpetraCrsMatrix::getGlobalNumEntries"); return mtx_->getGlobalNumEntries(); }
#else
    template <class Scalar, class Node>
    global_size_t TpetraCrsMatrix<Scalar,Node>::getGlobalNumEntries() const { XPETRA_MONITOR("TpetraCrsMatrix::getGlobalNumEntries"); return mtx_->getGlobalNumEntries(); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    size_t TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getNodeNumEntries() const { XPETRA_MONITOR("TpetraCrsMatrix::getNodeNumEntries"); return mtx_->getNodeNumEntries(); }
#else
    template <class Scalar, class Node>
    size_t TpetraCrsMatrix<Scalar,Node>::getNodeNumEntries() const { XPETRA_MONITOR("TpetraCrsMatrix::getNodeNumEntries"); return mtx_->getNodeNumEntries(); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    size_t TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getNumEntriesInLocalRow(LocalOrdinal localRow) const { XPETRA_MONITOR("TpetraCrsMatrix::getNumEntriesInLocalRow"); return mtx_->getNumEntriesInLocalRow(localRow); }
#else
    template <class Scalar, class Node>
    size_t TpetraCrsMatrix<Scalar,Node>::getNumEntriesInLocalRow(LocalOrdinal localRow) const { XPETRA_MONITOR("TpetraCrsMatrix::getNumEntriesInLocalRow"); return mtx_->getNumEntriesInLocalRow(localRow); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    size_t TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getNumEntriesInGlobalRow(GlobalOrdinal globalRow) const { XPETRA_MONITOR("TpetraCrsMatrix::getNumEntriesInGlobalRow"); return mtx_->getNumEntriesInGlobalRow(globalRow); }
#else
    template <class Scalar, class Node>
    size_t TpetraCrsMatrix<Scalar,Node>::getNumEntriesInGlobalRow(GlobalOrdinal globalRow) const { XPETRA_MONITOR("TpetraCrsMatrix::getNumEntriesInGlobalRow"); return mtx_->getNumEntriesInGlobalRow(globalRow); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    size_t TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getGlobalMaxNumRowEntries() const { XPETRA_MONITOR("TpetraCrsMatrix::getGlobalMaxNumRowEntries"); return mtx_->getGlobalMaxNumRowEntries(); }
#else
    template <class Scalar, class Node>
    size_t TpetraCrsMatrix<Scalar,Node>::getGlobalMaxNumRowEntries() const { XPETRA_MONITOR("TpetraCrsMatrix::getGlobalMaxNumRowEntries"); return mtx_->getGlobalMaxNumRowEntries(); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    size_t TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getNodeMaxNumRowEntries() const { XPETRA_MONITOR("TpetraCrsMatrix::getNodeMaxNumRowEntries"); return mtx_->getNodeMaxNumRowEntries(); }
#else
    template <class Scalar, class Node>
    size_t TpetraCrsMatrix<Scalar,Node>::getNodeMaxNumRowEntries() const { XPETRA_MONITOR("TpetraCrsMatrix::getNodeMaxNumRowEntries"); return mtx_->getNodeMaxNumRowEntries(); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    bool TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::isLocallyIndexed() const { XPETRA_MONITOR("TpetraCrsMatrix::isLocallyIndexed"); return mtx_->isLocallyIndexed(); }
#else
    template <class Scalar, class Node>
    bool TpetraCrsMatrix<Scalar,Node>::isLocallyIndexed() const { XPETRA_MONITOR("TpetraCrsMatrix::isLocallyIndexed"); return mtx_->isLocallyIndexed(); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    bool TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::isGloballyIndexed() const { XPETRA_MONITOR("TpetraCrsMatrix::isGloballyIndexed"); return mtx_->isGloballyIndexed(); }
#else
    template <class Scalar, class Node>
    bool TpetraCrsMatrix<Scalar,Node>::isGloballyIndexed() const { XPETRA_MONITOR("TpetraCrsMatrix::isGloballyIndexed"); return mtx_->isGloballyIndexed(); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    bool TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::isFillComplete() const { XPETRA_MONITOR("TpetraCrsMatrix::isFillComplete"); return mtx_->isFillComplete(); }
#else
    template <class Scalar, class Node>
    bool TpetraCrsMatrix<Scalar,Node>::isFillComplete() const { XPETRA_MONITOR("TpetraCrsMatrix::isFillComplete"); return mtx_->isFillComplete(); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    bool TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::isFillActive() const { XPETRA_MONITOR("TpetraCrsMatrix::isFillActive"); return mtx_->isFillActive(); }
#else
    template <class Scalar, class Node>
    bool TpetraCrsMatrix<Scalar,Node>::isFillActive() const { XPETRA_MONITOR("TpetraCrsMatrix::isFillActive"); return mtx_->isFillActive(); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    typename ScalarTraits< Scalar >::magnitudeType TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getFrobeniusNorm() const { XPETRA_MONITOR("TpetraCrsMatrix::getFrobeniusNorm"); return mtx_->getFrobeniusNorm(); }
#else
    template <class Scalar, class Node>
    typename ScalarTraits< Scalar >::magnitudeType TpetraCrsMatrix<Scalar,Node>::getFrobeniusNorm() const { XPETRA_MONITOR("TpetraCrsMatrix::getFrobeniusNorm"); return mtx_->getFrobeniusNorm(); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    bool TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::supportsRowViews() const { XPETRA_MONITOR("TpetraCrsMatrix::supportsRowViews"); return mtx_->supportsRowViews(); }
#else
    template <class Scalar, class Node>
    bool TpetraCrsMatrix<Scalar,Node>::supportsRowViews() const { XPETRA_MONITOR("TpetraCrsMatrix::supportsRowViews"); return mtx_->supportsRowViews(); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getLocalRowCopy(LocalOrdinal LocalRow, const ArrayView< LocalOrdinal > &Indices, const ArrayView< Scalar > &Values, size_t &NumEntries) const { XPETRA_MONITOR("TpetraCrsMatrix::getLocalRowCopy"); mtx_->getLocalRowCopy(LocalRow, Indices, Values, NumEntries); }
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::getLocalRowCopy(LocalOrdinal LocalRow, const ArrayView< LocalOrdinal > &Indices, const ArrayView< Scalar > &Values, size_t &NumEntries) const { XPETRA_MONITOR("TpetraCrsMatrix::getLocalRowCopy"); mtx_->getLocalRowCopy(LocalRow, Indices, Values, NumEntries); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getGlobalRowView(GlobalOrdinal GlobalRow, ArrayView< const GlobalOrdinal > &indices, ArrayView< const Scalar > &values) const { XPETRA_MONITOR("TpetraCrsMatrix::getGlobalRowView"); mtx_->getGlobalRowView(GlobalRow, indices, values); }
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::getGlobalRowView(GlobalOrdinal GlobalRow, ArrayView< const GlobalOrdinal > &indices, ArrayView< const Scalar > &values) const { XPETRA_MONITOR("TpetraCrsMatrix::getGlobalRowView"); mtx_->getGlobalRowView(GlobalRow, indices, values); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getGlobalRowCopy(GlobalOrdinal GlobalRow, const ArrayView< GlobalOrdinal > &indices, const ArrayView< Scalar > &values, size_t &numEntries) const { XPETRA_MONITOR("TpetraCrsMatrix::getGlobalRowCopy"); mtx_->getGlobalRowCopy(GlobalRow, indices, values, numEntries); }
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::getGlobalRowCopy(GlobalOrdinal GlobalRow, const ArrayView< GlobalOrdinal > &indices, const ArrayView< Scalar > &values, size_t &numEntries) const { XPETRA_MONITOR("TpetraCrsMatrix::getGlobalRowCopy"); mtx_->getGlobalRowCopy(GlobalRow, indices, values, numEntries); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getLocalRowView(LocalOrdinal LocalRow, ArrayView< const LocalOrdinal > &indices, ArrayView< const Scalar > &values) const { XPETRA_MONITOR("TpetraCrsMatrix::getLocalRowView"); mtx_->getLocalRowView(LocalRow, indices, values); }
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::getLocalRowView(LocalOrdinal LocalRow, ArrayView< const LocalOrdinal > &indices, ArrayView< const Scalar > &values) const { XPETRA_MONITOR("TpetraCrsMatrix::getLocalRowView"); mtx_->getLocalRowView(LocalRow, indices, values); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::apply(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &X, MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &Y, Teuchos::ETransp mode, Scalar alpha, Scalar beta) const { XPETRA_MONITOR("TpetraCrsMatrix::apply"); mtx_->apply(toTpetra(X), toTpetra(Y), mode, alpha, beta); }
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::apply(const MultiVector< Scalar, Node > &X, MultiVector< Scalar, Node > &Y, Teuchos::ETransp mode, Scalar alpha, Scalar beta) const { XPETRA_MONITOR("TpetraCrsMatrix::apply"); mtx_->apply(toTpetra(X), toTpetra(Y), mode, alpha, beta); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getDomainMap() const { XPETRA_MONITOR("TpetraCrsMatrix::getDomainMap"); return toXpetra(mtx_->getDomainMap()); }
#else
    template <class Scalar, class Node>
    const RCP< const Map<Node > >  TpetraCrsMatrix<Scalar,Node>::getDomainMap() const { XPETRA_MONITOR("TpetraCrsMatrix::getDomainMap"); return toXpetra(mtx_->getDomainMap()); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getRangeMap() const { XPETRA_MONITOR("TpetraCrsMatrix::getRangeMap"); return toXpetra(mtx_->getRangeMap()); }
#else
    template <class Scalar, class Node>
    const RCP< const Map<Node > >  TpetraCrsMatrix<Scalar,Node>::getRangeMap() const { XPETRA_MONITOR("TpetraCrsMatrix::getRangeMap"); return toXpetra(mtx_->getRangeMap()); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    std::string TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::description() const { XPETRA_MONITOR("TpetraCrsMatrix::description"); return mtx_->description(); }
#else
    template <class Scalar, class Node>
    std::string TpetraCrsMatrix<Scalar,Node>::description() const { XPETRA_MONITOR("TpetraCrsMatrix::description"); return mtx_->description(); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel) const { XPETRA_MONITOR("TpetraCrsMatrix::describe"); mtx_->describe(out, verbLevel); }
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel) const { XPETRA_MONITOR("TpetraCrsMatrix::describe"); mtx_->describe(out, verbLevel); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::setObjectLabel( const std::string &objectLabel ) {
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::setObjectLabel( const std::string &objectLabel ) {
#endif
      XPETRA_MONITOR("TpetraCrsMatrix::setObjectLabel");
      Teuchos::LabeledObject::setObjectLabel(objectLabel);
      mtx_->setObjectLabel(objectLabel);
    }



#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsMatrix(const TpetraCrsMatrix& matrix)
      :  mtx_(Teuchos::rcp(new Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>(*(matrix.mtx_),Teuchos::Copy))) {}
#else
    template <class Scalar, class Node>
    TpetraCrsMatrix<Scalar,Node>::TpetraCrsMatrix(const TpetraCrsMatrix& matrix)
      :  mtx_(Teuchos::rcp(new Tpetra::CrsMatrix<Scalar,Node>(*(matrix.mtx_),Teuchos::Copy))) {}
#endif
      
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getLocalDiagCopy(Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &diag) const {
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::getLocalDiagCopy(Vector< Scalar, Node > &diag) const {
#endif
      XPETRA_MONITOR("TpetraCrsMatrix::getLocalDiagCopy");
      XPETRA_DYNAMIC_CAST(TpetraVectorClass, diag, tDiag, "Xpetra::TpetraCrsMatrix.getLocalDiagCopy() only accept Xpetra::TpetraVector as input arguments.");
      mtx_->getLocalDiagCopy(*tDiag.getTpetra_Vector());
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getLocalDiagOffsets(Teuchos::ArrayRCP<size_t> &offsets) const {
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::getLocalDiagOffsets(Teuchos::ArrayRCP<size_t> &offsets) const {
#endif
      XPETRA_MONITOR("TpetraCrsMatrix::getLocalDiagOffsets");
      mtx_->getLocalDiagOffsets(offsets);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getLocalDiagCopy(Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &diag, const Teuchos::ArrayView<const size_t> &offsets) const {
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::getLocalDiagCopy(Vector< Scalar, Node > &diag, const Teuchos::ArrayView<const size_t> &offsets) const {
#endif
      XPETRA_MONITOR("TpetraCrsMatrix::getLocalDiagCopy");
      mtx_->getLocalDiagCopy(*(toTpetra(diag)), offsets);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::replaceDiag(const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &diag) {
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::replaceDiag(const Vector<Scalar, Node> &diag) {
#endif
      XPETRA_MONITOR("TpetraCrsMatrix::replaceDiag");
      Tpetra::replaceDiagonalCrsMatrix(*mtx_, *(toTpetra(diag)));
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::leftScale (const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& x) {
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::leftScale (const Vector<Scalar, Node>& x) {
#endif
      XPETRA_MONITOR("TpetraCrsMatrix::leftScale");
      mtx_->leftScale(*(toTpetra(x)));
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::rightScale (const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& x) {
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::rightScale (const Vector<Scalar, Node>& x) {
#endif
      XPETRA_MONITOR("TpetraCrsMatrix::rightScale");
      mtx_->rightScale(*(toTpetra(x)));
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getMap() const { XPETRA_MONITOR("TpetraCrsMatrix::getMap"); return rcp( new TpetraMap< LocalOrdinal, GlobalOrdinal, Node >(mtx_->getMap()) ); }
#else
    template <class Scalar, class Node>
    Teuchos::RCP< const Map<Node > > TpetraCrsMatrix<Scalar,Node>::getMap() const { XPETRA_MONITOR("TpetraCrsMatrix::getMap"); return rcp( new TpetraMap<Node >(mtx_->getMap()) ); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::doImport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &source,
                  const Import< LocalOrdinal, GlobalOrdinal, Node > &importer, CombineMode CM) {
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::doImport(const DistObject<char,Node> &source,
                  const Import<Node > &importer, CombineMode CM) {
#endif
      XPETRA_MONITOR("TpetraCrsMatrix::doImport");

      XPETRA_DYNAMIC_CAST(const TpetraCrsMatrixClass, source, tSource, "Xpetra::TpetraCrsMatrix::doImport only accept Xpetra::TpetraCrsMatrix as input arguments.");//TODO: remove and use toTpetra()
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP< const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > v = tSource.getTpetra_CrsMatrix();
#else
      RCP< const Tpetra::CrsMatrix<Scalar, Node> > v = tSource.getTpetra_CrsMatrix();
#endif
      //mtx_->doImport(toTpetraCrsMatrix(source), *tImporter.getTpetra_Import(), toTpetra(CM));
      mtx_->doImport(*v, toTpetra(importer), toTpetra(CM));
    }

    //! Export.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::doExport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &dest,
                  const Import< LocalOrdinal, GlobalOrdinal, Node >& importer, CombineMode CM) {
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::doExport(const DistObject<char,Node> &dest,
                  const Import<Node >& importer, CombineMode CM) {
#endif
      XPETRA_MONITOR("TpetraCrsMatrix::doExport");

      XPETRA_DYNAMIC_CAST(const TpetraCrsMatrixClass, dest, tDest, "Xpetra::TpetraCrsMatrix::doImport only accept Xpetra::TpetraCrsMatrix as input arguments.");//TODO: remove and use toTpetra()
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP< const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > v = tDest.getTpetra_CrsMatrix();
#else
      RCP< const Tpetra::CrsMatrix<Scalar, Node> > v = tDest.getTpetra_CrsMatrix();
#endif
      mtx_->doExport(*v, toTpetra(importer), toTpetra(CM));

    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::doImport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &source,
                  const Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM) {
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::doImport(const DistObject<char,Node> &source,
                  const Export<Node >& exporter, CombineMode CM) {
#endif
      XPETRA_MONITOR("TpetraCrsMatrix::doImport");

      XPETRA_DYNAMIC_CAST(const TpetraCrsMatrixClass, source, tSource, "Xpetra::TpetraCrsMatrix::doImport only accept Xpetra::TpetraCrsMatrix as input arguments.");//TODO: remove and use toTpetra()
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP< const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > v = tSource.getTpetra_CrsMatrix();
#else
      RCP< const Tpetra::CrsMatrix<Scalar, Node> > v = tSource.getTpetra_CrsMatrix();
#endif
      mtx_->doImport(*v, toTpetra(exporter), toTpetra(CM));

    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::doExport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &dest,
                  const Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM) {
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::doExport(const DistObject<char,Node> &dest,
                  const Export<Node >& exporter, CombineMode CM) {
#endif
      XPETRA_MONITOR("TpetraCrsMatrix::doExport");

      XPETRA_DYNAMIC_CAST(const TpetraCrsMatrixClass, dest, tDest, "Xpetra::TpetraCrsMatrix::doImport only accept Xpetra::TpetraCrsMatrix as input arguments.");//TODO: remove and use toTpetra()
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP< const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > v = tDest.getTpetra_CrsMatrix();
#else
      RCP< const Tpetra::CrsMatrix<Scalar, Node> > v = tDest.getTpetra_CrsMatrix();
#endif
      mtx_->doExport(*v, toTpetra(exporter), toTpetra(CM));

    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::removeEmptyProcessesInPlace (const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> >& newMap) {
#else
    template <class Scalar, class Node>
    void TpetraCrsMatrix<Scalar,Node>::removeEmptyProcessesInPlace (const Teuchos::RCP<const Map<Node> >& newMap) {
#endif
      XPETRA_MONITOR("TpetraCrsMatrix::removeEmptyProcessesInPlace");
      mtx_->removeEmptyProcessesInPlace(toTpetra(newMap));
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    bool TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::hasMatrix() const {
#else
    template <class Scalar, class Node>
    bool TpetraCrsMatrix<Scalar,Node>::hasMatrix() const {
#endif
      return ! mtx_.is_null ();
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraCrsMatrix(const Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &mtx) : mtx_(mtx) {  }
#else
    template <class Scalar, class Node>
    TpetraCrsMatrix<Scalar,Node>::TpetraCrsMatrix(const Teuchos::RCP<Tpetra::CrsMatrix<Scalar, Node> > &mtx) : mtx_(mtx) {  }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    RCP<const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getTpetra_CrsMatrix() const { return mtx_; }
#else
    template <class Scalar, class Node>
    RCP<const Tpetra::CrsMatrix<Scalar, Node> > TpetraCrsMatrix<Scalar,Node>::getTpetra_CrsMatrix() const { return mtx_; }
#endif

    //! Get the underlying Tpetra matrix
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getTpetra_CrsMatrixNonConst() const { return mtx_; } //TODO: remove
#else
    template <class Scalar, class Node>
    RCP<Tpetra::CrsMatrix<Scalar, Node> > TpetraCrsMatrix<Scalar,Node>::getTpetra_CrsMatrixNonConst() const { return mtx_; } //TODO: remove
#endif

 //! Compute a residual R = B - (*this) * X
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::residual(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > & X,
                                                                         const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > & B,
                                                                         MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > & R) const { 
#else
  template <class Scalar, class Node>
  void TpetraCrsMatrix<Scalar,Node>::residual(const MultiVector< Scalar, Node > & X,
                                                                         const MultiVector< Scalar, Node > & B,
                                                                         MultiVector< Scalar, Node > & R) const { 
#endif
    Tpetra::Details::residual(*mtx_,toTpetra(X),toTpetra(B),toTpetra(R));
  }


////////////////////////////////////////////
////////////////////////////////////////////
// End of TpetrCrsMatrix class definition //
////////////////////////////////////////////
////////////////////////////////////////////

} // Xpetra namespace

#endif // XPETRA_TPETRACRSMATRIX_DEF_HPP
