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
#ifndef XPETRA_TPETRABLOCKCRSMATRIX_DEF_HPP
#define XPETRA_TPETRABLOCKCRSMATRIX_DEF_HPP

#include "Xpetra_TpetraBlockCrsMatrix_decl.hpp"

namespace Xpetra {


    //! Constructor specifying fixed number of entries for each row (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, 
#else
    template<class Scalar, class Node>
    TpetraBlockCrsMatrix<Scalar, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map<Node > > &rowMap, 
#endif
                         size_t maxNumEntriesPerRow, 
                         const Teuchos::RCP< Teuchos::ParameterList > &params)
    { 
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented"); 
    }


    //! Constructor specifying (possibly different) number of entries in each row (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, 
#else
    template<class Scalar, class Node>
    TpetraBlockCrsMatrix<Scalar, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map<Node > > &rowMap, 
#endif
                         const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, 
                         const Teuchos::RCP< Teuchos::ParameterList > &params)
    {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


    //! Constructor specifying column Map and fixed number of entries for each row (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, 
                         const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, 
#else
    template<class Scalar, class Node>
    TpetraBlockCrsMatrix<Scalar, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map<Node > > &rowMap, 
                         const Teuchos::RCP< const Map<Node > > &colMap, 
#endif
                         size_t maxNumEntriesPerRow, 
                         const Teuchos::RCP< Teuchos::ParameterList > &params)
    {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented"); 
    }


    //! Constructor specifying column Map and number of entries in each row (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, 
                         const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, 
#else
    template<class Scalar, class Node>
    TpetraBlockCrsMatrix<Scalar, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map<Node > > &rowMap, 
                         const Teuchos::RCP< const Map<Node > > &colMap, 
#endif
                         const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, 
                         const Teuchos::RCP< Teuchos::ParameterList > &params)
    { 
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


    //! Constructor specifying a previously constructed graph ( not implemented )
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP< const CrsGraph< LocalOrdinal, GlobalOrdinal, Node> > &graph, 
#else
    template<class Scalar, class Node>
    TpetraBlockCrsMatrix<Scalar, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP< const CrsGraph<Node> > &graph, 
#endif
                         const Teuchos::RCP< Teuchos::ParameterList > &params)
        // : mtx_(Teuchos::rcp(new Tpetra::BlockCrsMatrix< Scalar, LocalOrdinal, GlobalOrdinal, Node >(toTpetra(graph), params)))
        // * there is no Tpetra::BlockCrsMatrix(graph, params) c'tor.  We throw anyways here so no need to set mtx_.
    {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


    //! Constructor specifying a previously constructed graph & blocksize
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP< const CrsGraph< LocalOrdinal, GlobalOrdinal, Node> > &graph, 
#else
    template<class Scalar, class Node>
    TpetraBlockCrsMatrix<Scalar, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP< const CrsGraph<Node> > &graph, 
#endif
                         const LocalOrdinal blockSize)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      : mtx_(Teuchos::rcp(new Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(*toTpetra(graph), blockSize))) 
#else
      : mtx_(Teuchos::rcp(new Tpetra::BlockCrsMatrix<Scalar, Node>(*toTpetra(graph), blockSize))) 
#endif
    { }


    //! Constructor for a fused import ( not implemented )
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& sourceMatrix,
                         const Import<LocalOrdinal,GlobalOrdinal,Node> & importer,
                         const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap,
                         const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap,
#else
    template<class Scalar, class Node>
    TpetraBlockCrsMatrix<Scalar, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,Node> >& sourceMatrix,
                         const Import<Node> & importer,
                         const Teuchos::RCP<const Map<Node> >& domainMap,
                         const Teuchos::RCP<const Map<Node> >& rangeMap,
#endif
                         const Teuchos::RCP<Teuchos::ParameterList>& params)
    {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


    //! Constructor for a fused export (not implemented(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& sourceMatrix,
                         const Export<LocalOrdinal,GlobalOrdinal,Node> & exporter,
                         const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap,
                         const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap,
#else
    template<class Scalar, class Node>
    TpetraBlockCrsMatrix<Scalar, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,Node> >& sourceMatrix,
                         const Export<Node> & exporter,
                         const Teuchos::RCP<const Map<Node> >& domainMap,
                         const Teuchos::RCP<const Map<Node> >& rangeMap,
#endif
                         const Teuchos::RCP<Teuchos::ParameterList>& params)
    {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


    //! Constructor for a fused import ( not implemented )
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& sourceMatrix,
                         const Import<LocalOrdinal,GlobalOrdinal,Node> & RowImporter,
                         const Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Node> > DomainImporter,
                         const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap,
                         const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap,
#else
    template<class Scalar, class Node>
    TpetraBlockCrsMatrix<Scalar, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,Node> >& sourceMatrix,
                         const Import<Node> & RowImporter,
                         const Teuchos::RCP<const Import<Node> > DomainImporter,
                         const Teuchos::RCP<const Map<Node> >& domainMap,
                         const Teuchos::RCP<const Map<Node> >& rangeMap,
#endif
                         const Teuchos::RCP<Teuchos::ParameterList>& params)
    {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }
    

    //! Constructor for a fused export (not implemented(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& sourceMatrix,
                         const Export<LocalOrdinal,GlobalOrdinal,Node> & RowExporter,
                         const Teuchos::RCP<const Export<LocalOrdinal,GlobalOrdinal,Node> > DomainExporter,
                         const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap,
                         const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap,
#else
    template<class Scalar, class Node>
    TpetraBlockCrsMatrix<Scalar, Node>::
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,Node> >& sourceMatrix,
                         const Export<Node> & RowExporter,
                         const Teuchos::RCP<const Export<Node> > DomainExporter,
                         const Teuchos::RCP<const Map<Node> >& domainMap,
                         const Teuchos::RCP<const Map<Node> >& rangeMap,
#endif
                         const Teuchos::RCP<Teuchos::ParameterList>& params)
    {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


    //! Destructor.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    template<class Scalar, class Node>
    TpetraBlockCrsMatrix<Scalar, Node>::
#endif
    ~TpetraBlockCrsMatrix() {  }


    //@}


    //! Insert matrix entries, using global TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::IDs (not implemented)    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    TpetraBlockCrsMatrix<Scalar, Node>::
#endif
    insertGlobalValues(GlobalOrdinal globalRow, 
                       const ArrayView< const GlobalOrdinal > &cols, 
                       const ArrayView< const Scalar > &vals)
    {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


    //! Insert matrix entries, using local IDs (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    TpetraBlockCrsMatrix<Scalar, Node>::
#endif
    insertLocalValues(LocalOrdinal localRow, 
                      const ArrayView< const LocalOrdinal > &cols, 
                      const ArrayView< const Scalar > &vals)
    {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


    //! Replace matrix entries, using global IDs (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    TpetraBlockCrsMatrix<Scalar, Node>::
#endif
    replaceGlobalValues(GlobalOrdinal globalRow, 
                        const ArrayView< const GlobalOrdinal > &cols, 
                        const ArrayView< const Scalar > &vals)
    {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


    //! Replace matrix entries, using local IDs.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    TpetraBlockCrsMatrix<Scalar, Node>::
#endif
    replaceLocalValues (LocalOrdinal localRow,const ArrayView<const LocalOrdinal> &cols,const ArrayView<const Scalar> &vals)
    {
      XPETRA_MONITOR("TpetraBlockCrsMatrix::replaceLocalValues");
      mtx_->replaceLocalValues(localRow,cols.getRawPtr(),vals.getRawPtr(),cols.size());
    }


    //! Set all matrix entries equal to scalarThis.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    TpetraBlockCrsMatrix<Scalar, Node>::
#endif
    setAllToScalar(const Scalar &alpha) 
    { 
      XPETRA_MONITOR("TpetraBlockCrsMatrix::setAllToScalar"); mtx_->setAllToScalar(alpha); 
    }


    //! Scale the current values of a matrix, this = alpha*this (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    TpetraBlockCrsMatrix<Scalar, Node>::
#endif
    scale(const Scalar &alpha)
    {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


    //! Allocates and returns ArrayRCPs of the Crs arrays --- This is an Xpetra-only routine.
    //** \warning This is an expert-only routine and should not be called from user code. (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    TpetraBlockCrsMatrix<Scalar, Node>::
#endif
    allocateAllValues(size_t numNonZeros,ArrayRCP<size_t> & rowptr, ArrayRCP<LocalOrdinal> & colind, ArrayRCP<Scalar> & values)
    {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


    //! Sets the 1D pointer arrays of the graph (not impelmented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    TpetraBlockCrsMatrix<Scalar, Node>::
#endif
    setAllValues(const ArrayRCP<size_t> & rowptr, const ArrayRCP<LocalOrdinal> & colind, const ArrayRCP<Scalar> & values)
    {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


    //! Gets the 1D pointer arrays of the graph (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    TpetraBlockCrsMatrix<Scalar, Node>::
#endif
    getAllValues(ArrayRCP<const size_t>& rowptr, 
                 ArrayRCP<const LocalOrdinal>& colind, 
                 ArrayRCP<const Scalar>& values) const
    { 
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented"); 
    }


    //@}
   
    // Transformational Methods
    //@{


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    resumeFill(const RCP< ParameterList > &params)
    { 
      /*noop*/ 
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
    fillComplete(const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &domainMap, 
                 const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rangeMap, 
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
    fillComplete(const RCP< const Map<Node > > &domainMap, 
                 const RCP< const Map<Node > > &rangeMap, 
#endif
                 const RCP< ParameterList > &params)
    { 
      /*noop*/ 
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    fillComplete(const RCP< ParameterList > &params)
    { 
      /*noop*/ 
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
    replaceDomainMapAndImporter(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >& newDomainMap, 
                                Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Node> > & newImporter)
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
    replaceDomainMapAndImporter(const Teuchos::RCP< const Map<Node > >& newDomainMap, 
                                Teuchos::RCP<const Import<Node> > & newImporter)
#endif
    { 
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented"); 
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
    expertStaticFillComplete(const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & domainMap,
                             const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & rangeMap,
                             const RCP<const Import<LocalOrdinal,GlobalOrdinal,Node> > &importer,
                             const RCP<const Export<LocalOrdinal,GlobalOrdinal,Node> > &exporter,
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
    expertStaticFillComplete(const RCP<const Map<Node> > & domainMap,
                             const RCP<const Map<Node> > & rangeMap,
                             const RCP<const Import<Node> > &importer,
                             const RCP<const Export<Node> > &exporter,
#endif
                             const RCP<ParameterList> &params)
    { 
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }  

    //@}


    //! @name Methods implementing RowMatrix


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    template<class Scalar, class Node>
    const RCP< const Map<Node > >  
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getRowMap() const
    { 
      XPETRA_MONITOR("TpetraBlockCrsMatrix::getRowMap"); return toXpetra(mtx_->getRowMap()); 
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    template<class Scalar, class Node>
    const RCP< const Map<Node > >  
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getColMap() const
    { 
      XPETRA_MONITOR("TpetraBlockCrsMatrix::getColMap"); return toXpetra(mtx_->getColMap()); 
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    RCP< const CrsGraph< LocalOrdinal, GlobalOrdinal, Node> > 
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    template<class Scalar, class Node>
    RCP< const CrsGraph<Node> > 
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getCrsGraph() const
    {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }
    

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    global_size_t 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getGlobalNumRows() const
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::getGlobalNumRows"); return mtx_->getGlobalNumRows(); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    global_size_t 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getGlobalNumCols() const
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::getGlobalNumCols"); return mtx_->getGlobalNumCols(); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    size_t 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getNodeNumRows() const
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::getNodeNumRows"); return mtx_->getNodeNumRows(); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    size_t 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getNodeNumCols() const
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::getNodeNumCols"); return mtx_->getNodeNumCols(); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    global_size_t 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getGlobalNumEntries() const
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::getGlobalNumEntries"); return mtx_->getGlobalNumEntries(); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    size_t 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getNodeNumEntries() const
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::getNodeNumEntries"); return mtx_->getNodeNumEntries(); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    size_t 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getNumEntriesInLocalRow(LocalOrdinal localRow) const
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::getNumEntriesInLocalRow"); return mtx_->getNumEntriesInLocalRow(localRow); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    size_t 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getNumEntriesInGlobalRow(GlobalOrdinal globalRow) const
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::getNumEntriesInGlobalRow"); return mtx_->getNumEntriesInGlobalRow(globalRow); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    size_t TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getGlobalMaxNumRowEntries() const
#else
    template<class Scalar, class Node>
    size_t TpetraBlockCrsMatrix<Scalar,Node>::getGlobalMaxNumRowEntries() const
#endif
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::getGlobalMaxNumRowEntries"); return mtx_->getGlobalMaxNumRowEntries(); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    size_t TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getNodeMaxNumRowEntries() const
#else
    template<class Scalar, class Node>
    size_t TpetraBlockCrsMatrix<Scalar,Node>::getNodeMaxNumRowEntries() const
#endif
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::getNodeMaxNumRowEntries"); return mtx_->getNodeMaxNumRowEntries(); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    bool TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::isLocallyIndexed() const
#else
    template<class Scalar, class Node>
    bool TpetraBlockCrsMatrix<Scalar,Node>::isLocallyIndexed() const
#endif
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::isLocallyIndexed"); return mtx_->isLocallyIndexed(); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    bool TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::isGloballyIndexed() const
#else
    template<class Scalar, class Node>
    bool TpetraBlockCrsMatrix<Scalar,Node>::isGloballyIndexed() const
#endif
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::isGloballyIndexed"); return mtx_->isGloballyIndexed(); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    bool TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::isFillComplete() const
#else
    template<class Scalar, class Node>
    bool TpetraBlockCrsMatrix<Scalar,Node>::isFillComplete() const
#endif
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::isFillComplete"); return mtx_->isFillComplete(); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    bool TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::isFillActive() const
#else
    template<class Scalar, class Node>
    bool TpetraBlockCrsMatrix<Scalar,Node>::isFillActive() const
#endif
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::isFillActive"); return false; }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    typename ScalarTraits< Scalar >::magnitudeType TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getFrobeniusNorm() const
#else
    template<class Scalar, class Node>
    typename ScalarTraits< Scalar >::magnitudeType TpetraBlockCrsMatrix<Scalar,Node>::getFrobeniusNorm() const
#endif
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::getFrobeniusNorm"); return mtx_->getFrobeniusNorm(); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    bool TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::supportsRowViews() const
#else
    template<class Scalar, class Node>
    bool TpetraBlockCrsMatrix<Scalar,Node>::supportsRowViews() const
#endif
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::supportsRowViews"); return mtx_->supportsRowViews(); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getLocalRowCopy(LocalOrdinal LocalRow, 
                    const ArrayView< LocalOrdinal > &Indices, 
                    const ArrayView< Scalar > &Values, 
                    size_t &NumEntries) const
    { 
        XPETRA_MONITOR("TpetraBlockCrsMatrix::getLocalRowCopy"); 
        mtx_->getLocalRowCopy(LocalRow, Indices, Values, NumEntries); 
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getGlobalRowView(GlobalOrdinal GlobalRow, 
                     ArrayView< const GlobalOrdinal > &indices, 
                     ArrayView< const Scalar > &values) const
    { 
        XPETRA_MONITOR("TpetraBlockCrsMatrix::getGlobalRowView"); 
        mtx_->getGlobalRowView(GlobalRow, indices, values);
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getGlobalRowCopy(GlobalOrdinal GlobalRow, 
                     const ArrayView< GlobalOrdinal > &indices, 
                     const ArrayView< Scalar > &values, 
                     size_t &numEntries) const
    { 
        XPETRA_MONITOR("TpetraBlockCrsMatrix::getGlobalRowCopy"); 
        mtx_->getGlobalRowCopy(GlobalRow, indices, values, numEntries); 
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getLocalRowView(LocalOrdinal LocalRow, ArrayView< const LocalOrdinal > &indices, 
                    ArrayView< const Scalar > &values) const
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::getLocalRowView"); mtx_->getLocalRowView(LocalRow, indices, values); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    bool 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    haveGlobalConstants() const
    { return true; }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
    apply(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &X, 
          MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &Y, 
#else
    template<class Scalar, class Node>
    void TpetraBlockCrsMatrix<Scalar,Node>::
    apply(const MultiVector< Scalar, Node > &X, 
          MultiVector< Scalar, Node > &Y, 
#endif
          Teuchos::ETransp mode, 
          Scalar alpha, 
          Scalar beta) const
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::apply"); mtx_->apply(toTpetra(X), toTpetra(Y), mode, alpha, beta); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    template<class Scalar, class Node>
    const RCP< const Map<Node > >  
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getDomainMap() const
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::getDomainMap"); return toXpetra(mtx_->getDomainMap()); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    template<class Scalar, class Node>
    const RCP< const Map<Node > >  
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getRangeMap() const
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::getRangeMap"); return toXpetra(mtx_->getRangeMap()); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    std::string 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    description() const
    { XPETRA_MONITOR("TpetraBlockCrsMatrix::description"); return mtx_->description(); }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    describe(Teuchos::FancyOStream &out, 
             const Teuchos::EVerbosityLevel verbLevel) const
    { 
        XPETRA_MONITOR("TpetraBlockCrsMatrix::describe"); 
        mtx_->describe(out, verbLevel); 
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    setObjectLabel( const std::string &objectLabel )
    {
        XPETRA_MONITOR("TpetraCrsMatrix::setObjectLabel");
        Teuchos::LabeledObject::setObjectLabel(objectLabel);
        mtx_->setObjectLabel(objectLabel);
    }



#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
    getLocalDiagCopy(Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &diag) const
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
    getLocalDiagCopy(Vector< Scalar, Node > &diag) const
#endif
    {
        XPETRA_MONITOR("TpetraBlockCrsMatrix::getLocalDiagCopy");
        XPETRA_DYNAMIC_CAST(TpetraVectorClass, 
                            diag, 
                            tDiag, 
                            "Xpetra::TpetraBlockCrsMatrix.getLocalDiagCopy() only accept Xpetra::TpetraVector as input arguments.");
        mtx_->getLocalDiagCopy(*tDiag.getTpetra_Vector());
    }


    //! Get a copy of the diagonal entries owned by this node, with local row indices.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
    getLocalDiagCopy(Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &diag, 
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
    getLocalDiagCopy(Vector< Scalar, Node > &diag, 
#endif
                     const Teuchos::ArrayView<const size_t> &offsets) const
    {
        throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }



#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getLocalDiagOffsets(Teuchos::ArrayRCP<size_t> &offsets) const
    {
        XPETRA_MONITOR("TpetraBlockCrsMatrix::getLocalDiagOffsets");

        const size_t lclNumRows = mtx_->getGraph()->getNodeNumRows();
        if (static_cast<size_t>(offsets.size()) < lclNumRows) 
        {
            offsets.resize(lclNumRows);
        }

        // The input ArrayRCP must always be a host pointer.  Thus, if
        // device_type::memory_space is Kokkos::HostSpace, it's OK for us
        // to write to that allocation directly as a Kokkos::View.
        typedef typename Node::device_type device_type;
        typedef typename device_type::memory_space memory_space;
        if (std::is_same<memory_space, Kokkos::HostSpace>::value) 
        {
            // It is always syntactically correct to assign a raw host
            // pointer to a device View, so this code will compile correctly
            // even if this branch never runs.
            typedef Kokkos::View<size_t*, device_type, Kokkos::MemoryUnmanaged> output_type;
            output_type offsetsOut (offsets.getRawPtr(), offsets.size());
            mtx_->getLocalDiagOffsets(offsetsOut);
        }
        else 
        {
            Kokkos::View<size_t*, device_type> offsetsTmp ("diagOffsets", offsets.size());
            mtx_->getLocalDiagOffsets(offsetsTmp);
            typedef Kokkos::View<size_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> output_type;
            output_type offsetsOut(offsets.getRawPtr(), offsets.size());
            Kokkos::deep_copy(offsetsOut, offsetsTmp);
        }
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
    replaceDiag(const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &diag)
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
    replaceDiag(const Vector<Scalar, Node> &diag)
#endif
    {
        throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix::replaceDiag: function not implemented");
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
    leftScale (const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& x)
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
    leftScale (const Vector<Scalar, Node>& x)
#endif
    {
        throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
    rightScale (const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& x)
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
    rightScale (const Vector<Scalar, Node>& x)
#endif
    {
        throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > 
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
    template<class Scalar, class Node>
    Teuchos::RCP< const Map<Node > > 
    TpetraBlockCrsMatrix<Scalar,Node>::
#endif
    getMap() const
    { 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        XPETRA_MONITOR("TpetraBlockCrsMatrix::getMap"); return rcp( new TpetraMap< LocalOrdinal, GlobalOrdinal, Node >(mtx_->getMap()) ); 
#else
        XPETRA_MONITOR("TpetraBlockCrsMatrix::getMap"); return rcp( new TpetraMap<Node >(mtx_->getMap()) ); 
#endif
    }


    //! Import.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
    doImport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &source,
             const Import< LocalOrdinal, GlobalOrdinal, Node > &importer, CombineMode CM)
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
    doImport(const DistObject<char,Node> &source,
             const Import<Node > &importer, CombineMode CM)
#endif
    {
        throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


    //! Export.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
    doExport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &dest,
                  const Import< LocalOrdinal, GlobalOrdinal, Node >& importer, CombineMode CM)
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
    doExport(const DistObject<char,Node> &dest,
                  const Import<Node >& importer, CombineMode CM)
#endif
    {
        throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


    //! Import (using an Exporter).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
    doImport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &source,
                  const Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM)
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
    doImport(const DistObject<char,Node> &source,
                  const Export<Node >& exporter, CombineMode CM)
#endif
    {   
        throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


    //! Export (using an Importer).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
    doExport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &dest,
                  const Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM)
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
    doExport(const DistObject<char,Node> &dest,
                  const Export<Node >& exporter, CombineMode CM)
#endif
    {
        throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template<class Scalar, class Node>
#endif
    void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
    removeEmptyProcessesInPlace (const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> >& newMap)
#else
    TpetraBlockCrsMatrix<Scalar,Node>::
    removeEmptyProcessesInPlace (const Teuchos::RCP<const Map<Node> >& newMap)
#endif
    {
        throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix function not implemented");
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
bool 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
TpetraBlockCrsMatrix<Scalar,Node>::
#endif
hasMatrix() const
{ 
    return !mtx_.is_null();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
TpetraBlockCrsMatrix(const Teuchos::RCP<Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &mtx) 
#else
template<class Scalar, class Node>
TpetraBlockCrsMatrix<Scalar,Node>::
TpetraBlockCrsMatrix(const Teuchos::RCP<Tpetra::BlockCrsMatrix<Scalar, Node> > &mtx) 
#endif
: mtx_(mtx)
{  }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > 
TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
template<class Scalar, class Node>
RCP<const Tpetra::BlockCrsMatrix<Scalar, Node> > 
TpetraBlockCrsMatrix<Scalar,Node>::
#endif
getTpetra_BlockCrsMatrix() const
{ 
    return mtx_; 
}


// TODO: remove
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > 
TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
template<class Scalar, class Node>
RCP<Tpetra::BlockCrsMatrix<Scalar, Node> > 
TpetraBlockCrsMatrix<Scalar,Node>::
#endif
getTpetra_BlockCrsMatrixNonConst() const
{ 
    return mtx_; 
} 

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef HAVE_XPETRA_TPETRA

// was:     typedef typename Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type local_matrix_type;
//using local_matrix_type = typename CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
typename CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::local_matrix_type
TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
template<class Scalar, class Node>
typename CrsMatrix<Scalar,Node>::local_matrix_type
TpetraBlockCrsMatrix<Scalar,Node>::
#endif
getLocalMatrix () const
{
    throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix does not support getLocalMatrix due to missing Kokkos::CrsMatrix in Tpetra's experimental implementation");

#ifndef __NVCC__
    local_matrix_type ret;
#endif  // __NVCC__

    TEUCHOS_UNREACHABLE_RETURN(ret);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Scalar, class Node>
#endif
void 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
TpetraBlockCrsMatrix<Scalar,Node>::
#endif
setAllValues (const typename local_matrix_type::row_map_type& ptr,
              const typename local_matrix_type::StaticCrsGraphType::entries_type::non_const_type& ind,
              const typename local_matrix_type::values_type& val)
{
    throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix does not support setAllValues due to missing Kokkos::CrsMatrix in Tpetra's experimental implementation");
}

#endif  // HAVE_XPETRA_TPETRA
#endif  // HAVE_XPETRA_KOKKOS_REFACTOR


#ifdef HAVE_XPETRA_EPETRA

#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))


  // specialization of TpetraBlockCrsMatrix for GO=LO=int and Node=EpetraNode
  template <class Scalar>
  class TpetraBlockCrsMatrix<Scalar,int,int,EpetraNode>
    : public CrsMatrix<Scalar,int,int,EpetraNode>//, public TpetraRowMatrix<Scalar,int,int,Node>
  {

    // The following typedef are used by the XPETRA_DYNAMIC_CAST() macro.
    typedef int LocalOrdinal;
    typedef int GlobalOrdinal;
    typedef EpetraNode Node;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> TpetraBlockCrsMatrixClass;
    typedef TpetraVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> TpetraVectorClass;
    typedef TpetraImport<LocalOrdinal,GlobalOrdinal,Node> TpetraImportClass;
    typedef TpetraExport<LocalOrdinal,GlobalOrdinal,Node> TpetraExportClass;
#else
    typedef TpetraBlockCrsMatrix<Scalar,Node> TpetraBlockCrsMatrixClass;
    typedef TpetraVector<Scalar,Node> TpetraVectorClass;
    typedef TpetraImport<Node> TpetraImportClass;
    typedef TpetraExport<Node> TpetraExportClass;
#endif

  public:

    //! @name Constructor/Destructor Methods

    //! Constructor specifying fixed number of entries for each row (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null) {
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map<Node > > &rowMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Constructor specifying (possibly different) number of entries in each row (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null) {
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map<Node > > &rowMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Constructor specifying column Map and fixed number of entries for each row (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null) {
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map<Node > > &rowMap, const Teuchos::RCP< const Map<Node > > &colMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Constructor specifying column Map and number of entries in each row (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null) {
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map<Node > > &rowMap, const Teuchos::RCP< const Map<Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Constructor specifying a previously constructed graph ( not implemented )
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP< const CrsGraph< LocalOrdinal, GlobalOrdinal, Node> > &graph, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null) {
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP< const CrsGraph<Node> > &graph, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Constructor specifying a previously constructed graph & blocksize
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP< const CrsGraph< LocalOrdinal, GlobalOrdinal, Node> > &graph, const LocalOrdinal blockSize) {
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP< const CrsGraph<Node> > &graph, const LocalOrdinal blockSize) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Constructor for a fused import ( not implemented )
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& sourceMatrix,
                    const Import<LocalOrdinal,GlobalOrdinal,Node> & importer,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap = Teuchos::null,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap = Teuchos::null,
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,Node> >& sourceMatrix,
                    const Import<Node> & importer,
                    const Teuchos::RCP<const Map<Node> >& domainMap = Teuchos::null,
                    const Teuchos::RCP<const Map<Node> >& rangeMap = Teuchos::null,
#endif
       const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null)
    { XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );}

    //! Constructor for a fused export (not implemented(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& sourceMatrix,
                    const Export<LocalOrdinal,GlobalOrdinal,Node> & exporter,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap = Teuchos::null,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap = Teuchos::null,
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,Node> >& sourceMatrix,
                    const Export<Node> & exporter,
                    const Teuchos::RCP<const Map<Node> >& domainMap = Teuchos::null,
                    const Teuchos::RCP<const Map<Node> >& rangeMap = Teuchos::null,
#endif
                    const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null)
    { XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );}

    //! Constructor for a fused import ( not implemented )
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& sourceMatrix,
                    const Import<LocalOrdinal,GlobalOrdinal,Node> & RowImporter,
                    const Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Node> > DomainImporter,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap,
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,Node> >& sourceMatrix,
                    const Import<Node> & RowImporter,
                    const Teuchos::RCP<const Import<Node> > DomainImporter,
                    const Teuchos::RCP<const Map<Node> >& domainMap,
                    const Teuchos::RCP<const Map<Node> >& rangeMap,
#endif
       const Teuchos::RCP<Teuchos::ParameterList>& params)
    { XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );}

    //! Constructor for a fused export (not implemented(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& sourceMatrix,
                    const Export<LocalOrdinal,GlobalOrdinal,Node> & RowExporter,
                    const Teuchos::RCP<const Export<LocalOrdinal,GlobalOrdinal,Node> > DomainExporter,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap,
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,Node> >& sourceMatrix,
                    const Export<Node> & RowExporter,
                    const Teuchos::RCP<const Export<Node> > DomainExporter,
                    const Teuchos::RCP<const Map<Node> >& domainMap,
                    const Teuchos::RCP<const Map<Node> >& rangeMap,
#endif
                    const Teuchos::RCP<Teuchos::ParameterList>& params)
    { 
        XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Destructor.
    ~TpetraBlockCrsMatrix() {  }


    //! @name Insertion/Removal Methods

    //! Insert matrix entries, using global IDs (not implemented)
    void insertGlobalValues(GlobalOrdinal globalRow, const ArrayView< const GlobalOrdinal > &cols, const ArrayView< const Scalar > &vals)
    {}

    //! Insert matrix entries, using local IDs (not implemented)
    void insertLocalValues(LocalOrdinal localRow, const ArrayView< const LocalOrdinal > &cols, const ArrayView< const Scalar > &vals)
    {}

    //! Replace matrix entries, using global IDs (not implemented)
    void replaceGlobalValues(GlobalOrdinal globalRow, const ArrayView< const GlobalOrdinal > &cols, const ArrayView< const Scalar > &vals)
    {}

    //! Replace matrix entries, using local IDs.
    void replaceLocalValues (LocalOrdinal localRow,const ArrayView<const LocalOrdinal> &cols,const ArrayView<const Scalar> &vals)
    {}

    //! Set all matrix entries equal to scalarThis.
    void setAllToScalar(const Scalar &alpha) {}

    //! Scale the current values of a matrix, this = alpha*this (not implemented)
    void scale(const Scalar &alpha)
    {}

    //! Allocates and returns ArrayRCPs of the Crs arrays --- This is an Xpetra-only routine.
    //** \warning This is an expert-only routine and should not be called from user code. (not implemented)
    void allocateAllValues(size_t numNonZeros,ArrayRCP<size_t> & rowptr, ArrayRCP<LocalOrdinal> & colind, ArrayRCP<Scalar> & values)
    {}

    //! Sets the 1D pointer arrays of the graph (not impelmented)
    void setAllValues(const ArrayRCP<size_t> & rowptr, const ArrayRCP<LocalOrdinal> & colind, const ArrayRCP<Scalar> & values)
    {}

    //! Gets the 1D pointer arrays of the graph (not implemented)
    void getAllValues(ArrayRCP<const size_t>& rowptr, ArrayRCP<const LocalOrdinal>& colind, ArrayRCP<const Scalar>& values) const
    {}


    //! @name Transformational Methods

    //!
    void resumeFill(const RCP< ParameterList > &params=null) { /*noop*/ }

    //! Signal that data entry is complete, specifying domain and range maps.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void fillComplete(const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &domainMap, const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rangeMap, const RCP< ParameterList > &params=null) { /*noop*/ }
#else
    void fillComplete(const RCP< const Map<Node > > &domainMap, const RCP< const Map<Node > > &rangeMap, const RCP< ParameterList > &params=null) { /*noop*/ }
#endif

    //! Signal that data entry is complete.
    void fillComplete(const RCP< ParameterList > &params=null) { /*noop*/ }


    //!  Replaces the current domainMap and importer with the user-specified objects.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void replaceDomainMapAndImporter(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >& newDomainMap, Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Node> >  & newImporter)
#else
    void replaceDomainMapAndImporter(const Teuchos::RCP< const Map<Node > >& newDomainMap, Teuchos::RCP<const Import<Node> >  & newImporter)
#endif
    {}

    //! Expert static fill complete
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void expertStaticFillComplete(const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & domainMap,
                                  const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & rangeMap,
                                  const RCP<const Import<LocalOrdinal,GlobalOrdinal,Node> > &importer=Teuchos::null,
                                  const RCP<const Export<LocalOrdinal,GlobalOrdinal,Node> > &exporter=Teuchos::null,
#else
    void expertStaticFillComplete(const RCP<const Map<Node> > & domainMap,
                                  const RCP<const Map<Node> > & rangeMap,
                                  const RCP<const Import<Node> > &importer=Teuchos::null,
                                  const RCP<const Export<Node> > &exporter=Teuchos::null,
#endif
                                  const RCP<ParameterList> &params=Teuchos::null)
    {}


    //! @name Methods implementing RowMatrix

    //! Returns the Map that describes the row distribution in this matrix.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  getRowMap() const { return Teuchos::null; }
#else
    const RCP< const Map<Node > >  getRowMap() const { return Teuchos::null; }
#endif

    //! Returns the Map that describes the column distribution in this matrix.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  getColMap() const { return Teuchos::null; }
#else
    const RCP< const Map<Node > >  getColMap() const { return Teuchos::null; }
#endif

    //! Returns the CrsGraph associated with this matrix.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const CrsGraph< LocalOrdinal, GlobalOrdinal, Node> > getCrsGraph() const
#else
    RCP< const CrsGraph<Node> > getCrsGraph() const
#endif
    {return Teuchos::null;}

    //! Number of global elements in the row map of this matrix.
    global_size_t getGlobalNumRows() const { return 0; }

    //! Number of global columns in the matrix.
    global_size_t getGlobalNumCols() const { return 0; }

    //! Returns the number of matrix rows owned on the calling node.
    size_t getNodeNumRows() const { return 0; }

    //! Returns the number of columns connected to the locally owned rows of this matrix.
    size_t getNodeNumCols() const { return 0; }

    //! Returns the global number of entries in this matrix.
    global_size_t getGlobalNumEntries() const { return 0; }

    //! Returns the local number of entries in this matrix.
    size_t getNodeNumEntries() const { return 0; }

    //! Returns the current number of entries on this node in the specified local row.
    size_t getNumEntriesInLocalRow(LocalOrdinal localRow) const { return 0; }

    //! Returns the current number of entries in the (locally owned) global row.
    size_t getNumEntriesInGlobalRow(GlobalOrdinal globalRow) const { return 0; }

    //! Returns the maximum number of entries across all rows/columns on all nodes.
    size_t getGlobalMaxNumRowEntries() const { return 0; }

    //! Returns the maximum number of entries across all rows/columns on this node.
    size_t getNodeMaxNumRowEntries() const { return 0; }

    //! If matrix indices are in the local range, this function returns true. Otherwise, this function returns false.
    bool isLocallyIndexed() const { return false; }

    //! If matrix indices are in the global range, this function returns true. Otherwise, this function returns false.
    bool isGloballyIndexed() const { return false; }

    //! Returns true if the matrix is in compute mode, i.e. if fillComplete() has been called.
    bool isFillComplete() const { return false; }

    //! Returns true if the matrix is in edit mode.
    bool isFillActive() const { return false; }

    //! Returns the Frobenius norm of the matrix.
    typename ScalarTraits< Scalar >::magnitudeType getFrobeniusNorm() const { return Teuchos::ScalarTraits<Scalar>::magnitude(Teuchos::ScalarTraits<Scalar>::zero()); }

    //! Returns true if getLocalRowView() and getGlobalRowView() are valid for this class.
    bool supportsRowViews() const { return false; }

    //! Extract a list of entries in a specified local row of the matrix. Put into storage allocated by calling routine.
    void getLocalRowCopy(LocalOrdinal LocalRow, const ArrayView< LocalOrdinal > &Indices, const ArrayView< Scalar > &Values, size_t &NumEntries) const {  }

    //! Extract a const, non-persisting view of global indices in a specified row of the matrix.
    void getGlobalRowView(GlobalOrdinal GlobalRow, ArrayView< const GlobalOrdinal > &indices, ArrayView< const Scalar > &values) const {  }

    //! Extract a list of entries in a specified global row of this matrix. Put into pre-allocated storage.
    void getGlobalRowCopy(GlobalOrdinal GlobalRow, const ArrayView< GlobalOrdinal > &indices, const ArrayView< Scalar > &values, size_t &numEntries) const {  }

    //! Extract a const, non-persisting view of local indices in a specified row of the matrix.
    void getLocalRowView(LocalOrdinal LocalRow, ArrayView< const LocalOrdinal > &indices, ArrayView< const Scalar > &values) const {  }

    //! Returns true if globalConstants have been computed; false otherwise
    bool haveGlobalConstants() const {return false;}


    //! @name Methods implementing Operator

    //! Computes the sparse matrix-multivector multiplication.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void apply(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &X, MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &Y, Teuchos::ETransp mode=Teuchos::NO_TRANS, Scalar alpha=ScalarTraits< Scalar >::one(), Scalar beta=ScalarTraits< Scalar >::zero()) const {  }
#else
    void apply(const MultiVector< Scalar, Node > &X, MultiVector< Scalar, Node > &Y, Teuchos::ETransp mode=Teuchos::NO_TRANS, Scalar alpha=ScalarTraits< Scalar >::one(), Scalar beta=ScalarTraits< Scalar >::zero()) const {  }
#endif

    //! Returns the Map associated with the domain of this operator. This will be null until fillComplete() is called.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  getDomainMap() const { return Teuchos::null; }
#else
    const RCP< const Map<Node > >  getDomainMap() const { return Teuchos::null; }
#endif

    //!
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  getRangeMap() const { return Teuchos::null; }
#else
    const RCP< const Map<Node > >  getRangeMap() const { return Teuchos::null; }
#endif


    //! @name Overridden from Teuchos::Describable

    //! A simple one-line description of this object.
    std::string description() const { return std::string(""); }

    //! Print the object with some verbosity level to an FancyOStream object.
    void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel=Teuchos::Describable::verbLevel_default) const {  }


    //! Deep copy constructor
    TpetraBlockCrsMatrix(const TpetraBlockCrsMatrix& matrix) {}

    //! Get a copy of the diagonal entries owned by this node, with local row idices 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void getLocalDiagCopy(Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &diag) const {    }
#else
    void getLocalDiagCopy(Vector< Scalar, Node > &diag) const {    }
#endif

    //! Get offsets of the diagonal entries in the matrix.
    void getLocalDiagOffsets(Teuchos::ArrayRCP<size_t> &offsets) const {    }

    //! Get a copy of the diagonal entries owned by this node, with local row indices.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void getLocalDiagCopy(Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &diag, const Teuchos::ArrayView<const size_t> &offsets) const
#else
    void getLocalDiagCopy(Vector< Scalar, Node > &diag, const Teuchos::ArrayView<const size_t> &offsets) const
#endif
    {}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void replaceDiag(const Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &diag) {    }
#else
    void replaceDiag(const Vector< Scalar, Node > &diag) {    }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void leftScale (const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& x) { }
    void rightScale (const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& x) { }
#else
    void leftScale (const Vector<Scalar, Node>& x) { }
    void rightScale (const Vector<Scalar, Node>& x) { }
#endif


    //! Implements DistObject interface

    //! Access function for the Tpetra::Map this DistObject was constructed with.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > getMap() const { return Teuchos::null; }
#else
    Teuchos::RCP< const Map<Node > > getMap() const { return Teuchos::null; }
#endif

    //! Import.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doImport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &source,
                  const Import< LocalOrdinal, GlobalOrdinal, Node > &importer, CombineMode CM)
#else
    void doImport(const DistObject<char,Node> &source,
                  const Import<Node > &importer, CombineMode CM)
#endif
    {}

    //! Export.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doExport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &dest,
                  const Import< LocalOrdinal, GlobalOrdinal, Node >& importer, CombineMode CM)
#else
    void doExport(const DistObject<char,Node> &dest,
                  const Import<Node >& importer, CombineMode CM)
#endif
    {}

    //! Import (using an Exporter).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doImport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &source,
                  const Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM)
#else
    void doImport(const DistObject<char,Node> &source,
                  const Export<Node >& exporter, CombineMode CM)
#endif
    {}

    //! Export (using an Importer).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doExport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &dest,
                  const Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM)
#else
    void doExport(const DistObject<char,Node> &dest,
                  const Export<Node >& exporter, CombineMode CM)
#endif
    {}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void removeEmptyProcessesInPlace (const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> >& newMap)
#else
    void removeEmptyProcessesInPlace (const Teuchos::RCP<const Map<Node> >& newMap)
#endif
    {}



    //! @name Xpetra specific

    //! Does this have an underlying matrix
    bool hasMatrix() const { return false; }

    //! TpetraBlockCrsMatrix constructor to wrap a Tpetra::BlockCrsMatrix object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP<Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &mtx) {
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP<Tpetra::BlockCrsMatrix<Scalar, Node> > &mtx) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Get the underlying Tpetra matrix
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<const Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > getTpetra_BlockCrsMatrix() const { return Teuchos::null; }
#else
    RCP<const Tpetra::BlockCrsMatrix<Scalar, Node> > getTpetra_BlockCrsMatrix() const { return Teuchos::null; }
#endif

    //! Get the underlying Tpetra matrix
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > getTpetra_BlockCrsMatrixNonConst() const { return Teuchos::null; }
#else
    RCP<Tpetra::BlockCrsMatrix<Scalar, Node> > getTpetra_BlockCrsMatrixNonConst() const { return Teuchos::null; }
#endif

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef HAVE_XPETRA_TPETRA
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef typename Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type local_matrix_type;
#else
    typedef typename Xpetra::CrsMatrix<Scalar, Node>::local_matrix_type local_matrix_type;
#endif

    local_matrix_type getLocalMatrix () const {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix does not support getLocalMatrix due to missing Kokkos::CrsMatrix in Tpetra's experimental implementation");
      local_matrix_type ret;
      return ret; // make compiler happy
    }

    void setAllValues (const typename local_matrix_type::row_map_type& ptr,
                       const typename local_matrix_type::StaticCrsGraphType::entries_type::non_const_type& ind,
                       const typename local_matrix_type::values_type& val)
    {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix does not support setAllValues due to missing Kokkos::CrsMatrix in Tpetra's experimental implementation");
    }
#endif  // HAVE_XPETRA_TPETRA
#endif  // HAVE_XPETRA_KOKKOS_REFACTOR

    }; // TpetraBlockCrsMatrix class


#endif  // #if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) 




#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_LONG_LONG))))

  // specialization of TpetraBlockCrsMatrix for GO=long long and Node=EpetraNode
  template <class Scalar>
  class TpetraBlockCrsMatrix<Scalar,int,long long,EpetraNode>
    : public CrsMatrix<Scalar,int,long long,EpetraNode>//, public TpetraRowMatrix<Scalar,int,int,Node>
  {

    // The following typedef are used by the XPETRA_DYNAMIC_CAST() macro.
    typedef int LocalOrdinal;
    typedef long long GlobalOrdinal;
    typedef EpetraNode Node;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> TpetraBlockCrsMatrixClass;
    typedef TpetraVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> TpetraVectorClass;
    typedef TpetraImport<LocalOrdinal,GlobalOrdinal,Node> TpetraImportClass;
    typedef TpetraExport<LocalOrdinal,GlobalOrdinal,Node> TpetraExportClass;
#else
    typedef TpetraBlockCrsMatrix<Scalar,Node> TpetraBlockCrsMatrixClass;
    typedef TpetraVector<Scalar,Node> TpetraVectorClass;
    typedef TpetraImport<Node> TpetraImportClass;
    typedef TpetraExport<Node> TpetraExportClass;
#endif

  public:

    //! @name Constructor/Destructor Methods

    //! Constructor specifying fixed number of entries for each row (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null)
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map<Node > > &rowMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null)
#endif
    {XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );}

    //! Constructor specifying (possibly different) number of entries in each row (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null)
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map<Node > > &rowMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null)
#endif
    {XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );}

    //! Constructor specifying column Map and fixed number of entries for each row (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null)
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map<Node > > &rowMap, const Teuchos::RCP< const Map<Node > > &colMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null)
#endif
    {XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );}

    //! Constructor specifying column Map and number of entries in each row (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null)
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP< const Map<Node > > &rowMap, const Teuchos::RCP< const Map<Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null)
#endif
    {XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );}

    //! Constructor specifying a previously constructed graph ( not implemented )
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP< const CrsGraph< LocalOrdinal, GlobalOrdinal, Node> > &graph, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null)
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP< const CrsGraph<Node> > &graph, const Teuchos::RCP< Teuchos::ParameterList > &params=Teuchos::null)
#endif
    {XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );}

    //! Constructor specifying a previously constructed graph & blocksize
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP< const CrsGraph< LocalOrdinal, GlobalOrdinal, Node> > &graph, const LocalOrdinal blockSize)
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP< const CrsGraph<Node> > &graph, const LocalOrdinal blockSize)
#endif
    {XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );}




    //! Constructor for a fused import (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& sourceMatrix,
                    const Import<LocalOrdinal,GlobalOrdinal,Node> & importer,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap = Teuchos::null,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap = Teuchos::null,
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,Node> >& sourceMatrix,
                    const Import<Node> & importer,
                    const Teuchos::RCP<const Map<Node> >& domainMap = Teuchos::null,
                    const Teuchos::RCP<const Map<Node> >& rangeMap = Teuchos::null,
#endif
       const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null)
    {XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );}

    //! Constructor for a fused export (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& sourceMatrix,
                    const Export<LocalOrdinal,GlobalOrdinal,Node> & exporter,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap = Teuchos::null,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap = Teuchos::null,
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,Node> >& sourceMatrix,
                    const Export<Node> & exporter,
                    const Teuchos::RCP<const Map<Node> >& domainMap = Teuchos::null,
                    const Teuchos::RCP<const Map<Node> >& rangeMap = Teuchos::null,
#endif
                    const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null)
    {XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );}

    //! Constructor for a fused import (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& sourceMatrix,
                    const Import<LocalOrdinal,GlobalOrdinal,Node> & RowImporter,
                    const Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Node> > DomainImporter,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap,
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,Node> >& sourceMatrix,
                    const Import<Node> & RowImporter,
                    const Teuchos::RCP<const Import<Node> > DomainImporter,
                    const Teuchos::RCP<const Map<Node> >& domainMap,
                    const Teuchos::RCP<const Map<Node> >& rangeMap,
#endif
       const Teuchos::RCP<Teuchos::ParameterList>& params)
    {XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );}

    //! Constructor for a fused export (not implemented)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& sourceMatrix,
                    const Export<LocalOrdinal,GlobalOrdinal,Node> & RowExporter,
                    const Teuchos::RCP<const Export<LocalOrdinal,GlobalOrdinal,Node> > DomainExporter,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap,
                    const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap,
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP<const Tpetra::BlockCrsMatrix<Scalar,Node> >& sourceMatrix,
                    const Export<Node> & RowExporter,
                    const Teuchos::RCP<const Export<Node> > DomainExporter,
                    const Teuchos::RCP<const Map<Node> >& domainMap,
                    const Teuchos::RCP<const Map<Node> >& rangeMap,
#endif
                    const Teuchos::RCP<Teuchos::ParameterList>& params)
    {XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );}

    //! Destructor.
    ~TpetraBlockCrsMatrix() {  }


    //! @name Insertion/Removal Methods

    //! Insert matrix entries, using global IDs (not implemented)
    void insertGlobalValues(GlobalOrdinal globalRow, const ArrayView< const GlobalOrdinal > &cols, const ArrayView< const Scalar > &vals)
    {}

    //! Insert matrix entries, using local IDs (not implemented)
    void insertLocalValues(LocalOrdinal localRow, const ArrayView< const LocalOrdinal > &cols, const ArrayView< const Scalar > &vals)
    {}

    //! Replace matrix entries, using global IDs (not implemented)
    void replaceGlobalValues(GlobalOrdinal globalRow, const ArrayView< const GlobalOrdinal > &cols, const ArrayView< const Scalar > &vals)
    {}

    //! Replace matrix entries, using local IDs.
    void replaceLocalValues (LocalOrdinal localRow,const ArrayView<const LocalOrdinal> &cols,const ArrayView<const Scalar> &vals)
    {}

    //! Set all matrix entries equal to scalarThis.
    void setAllToScalar(const Scalar &alpha) {}

    //! Scale the current values of a matrix, this = alpha*this (not implemented)
    void scale(const Scalar &alpha)
    {}

    //! Allocates and returns ArrayRCPs of the Crs arrays --- This is an Xpetra-only routine.
    //** \warning This is an expert-only routine and should not be called from user code. (not implemented)
    void allocateAllValues(size_t numNonZeros,ArrayRCP<size_t> & rowptr, ArrayRCP<LocalOrdinal> & colind, ArrayRCP<Scalar> & values)
    {}

    //! Sets the 1D pointer arrays of the graph (not impelmented)
    void setAllValues(const ArrayRCP<size_t> & rowptr, const ArrayRCP<LocalOrdinal> & colind, const ArrayRCP<Scalar> & values)
    {}

    //! Gets the 1D pointer arrays of the graph (not implemented)
    void getAllValues(ArrayRCP<const size_t>& rowptr, ArrayRCP<const LocalOrdinal>& colind, ArrayRCP<const Scalar>& values) const
    {}


    //! @name Transformational Methods

    //!
    void resumeFill(const RCP< ParameterList > &params=null) { /*noop*/ }

    //! Signal that data entry is complete, specifying domain and range maps.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void fillComplete(const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &domainMap, const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rangeMap, const RCP< ParameterList > &params=null) { /*noop*/ }
#else
    void fillComplete(const RCP< const Map<Node > > &domainMap, const RCP< const Map<Node > > &rangeMap, const RCP< ParameterList > &params=null) { /*noop*/ }
#endif

    //! Signal that data entry is complete.
    void fillComplete(const RCP< ParameterList > &params=null) { /*noop*/ }


    //!  Replaces the current domainMap and importer with the user-specified objects.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void replaceDomainMapAndImporter(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >& newDomainMap, Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Node> >  & newImporter)
#else
    void replaceDomainMapAndImporter(const Teuchos::RCP< const Map<Node > >& newDomainMap, Teuchos::RCP<const Import<Node> >  & newImporter)
#endif
    {}

    //! Expert static fill complete
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void expertStaticFillComplete(const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & domainMap,
                                  const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & rangeMap,
                                  const RCP<const Import<LocalOrdinal,GlobalOrdinal,Node> > &importer=Teuchos::null,
                                  const RCP<const Export<LocalOrdinal,GlobalOrdinal,Node> > &exporter=Teuchos::null,
#else
    void expertStaticFillComplete(const RCP<const Map<Node> > & domainMap,
                                  const RCP<const Map<Node> > & rangeMap,
                                  const RCP<const Import<Node> > &importer=Teuchos::null,
                                  const RCP<const Export<Node> > &exporter=Teuchos::null,
#endif
                                  const RCP<ParameterList> &params=Teuchos::null)
    {}


    //! @name Methods implementing RowMatrix

    //! Returns the Map that describes the row distribution in this matrix.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  getRowMap() const { return Teuchos::null; }
#else
    const RCP< const Map<Node > >  getRowMap() const { return Teuchos::null; }
#endif

    //! Returns the Map that describes the column distribution in this matrix.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  getColMap() const { return Teuchos::null; }
#else
    const RCP< const Map<Node > >  getColMap() const { return Teuchos::null; }
#endif

    //! Returns the CrsGraph associated with this matrix.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const CrsGraph< LocalOrdinal, GlobalOrdinal, Node> > getCrsGraph() const
#else
    RCP< const CrsGraph<Node> > getCrsGraph() const
#endif
    {return Teuchos::null;}

    //! Number of global elements in the row map of this matrix.
    global_size_t getGlobalNumRows() const { return 0; }

    //! Number of global columns in the matrix.
    global_size_t getGlobalNumCols() const { return 0; }

    //! Returns the number of matrix rows owned on the calling node.
    size_t getNodeNumRows() const { return 0; }

    //! Returns the number of columns connected to the locally owned rows of this matrix.
    size_t getNodeNumCols() const { return 0; }

    //! Returns the global number of entries in this matrix.
    global_size_t getGlobalNumEntries() const { return 0; }

    //! Returns the local number of entries in this matrix.
    size_t getNodeNumEntries() const { return 0; }

    //! Returns the current number of entries on this node in the specified local row.
    size_t getNumEntriesInLocalRow(LocalOrdinal localRow) const { return 0; }

    //! Returns the current number of entries in the (locally owned) global row.
    size_t getNumEntriesInGlobalRow(GlobalOrdinal globalRow) const { return 0; }

    //! Returns the maximum number of entries across all rows/columns on all nodes.
    size_t getGlobalMaxNumRowEntries() const { return 0; }

    //! Returns the maximum number of entries across all rows/columns on this node.
    size_t getNodeMaxNumRowEntries() const { return 0; }

    //! If matrix indices are in the local range, this function returns true. Otherwise, this function returns false.
    bool isLocallyIndexed() const { return false; }

    //! If matrix indices are in the global range, this function returns true. Otherwise, this function returns false.
    bool isGloballyIndexed() const { return false; }

    //! Returns true if the matrix is in compute mode, i.e. if fillComplete() has been called.
    bool isFillComplete() const { return false; }

    //! Returns true if the matrix is in edit mode.
    bool isFillActive() const { return false; }

    //! Returns the Frobenius norm of the matrix.
    typename ScalarTraits< Scalar >::magnitudeType getFrobeniusNorm() const { return Teuchos::ScalarTraits<Scalar>::magnitude(Teuchos::ScalarTraits<Scalar>::zero()); }

    //! Returns true if getLocalRowView() and getGlobalRowView() are valid for this class.
    bool supportsRowViews() const { return false; }

    //! Extract a list of entries in a specified local row of the matrix. Put into storage allocated by calling routine.
    void getLocalRowCopy(LocalOrdinal LocalRow, const ArrayView< LocalOrdinal > &Indices, const ArrayView< Scalar > &Values, size_t &NumEntries) const {  }

    //! Extract a const, non-persisting view of global indices in a specified row of the matrix.
    void getGlobalRowView(GlobalOrdinal GlobalRow, ArrayView< const GlobalOrdinal > &indices, ArrayView< const Scalar > &values) const {  }

    //! Extract a list of entries in a specified global row of this matrix. Put into pre-allocated storage.
    void getGlobalRowCopy(GlobalOrdinal GlobalRow, const ArrayView< GlobalOrdinal > &indices, const ArrayView< Scalar > &values, size_t &numEntries) const {  }

    //! Extract a const, non-persisting view of local indices in a specified row of the matrix.
    void getLocalRowView(LocalOrdinal LocalRow, ArrayView< const LocalOrdinal > &indices, ArrayView< const Scalar > &values) const {  }

    //! Returns true if globalConstants have been computed; false otherwise
    bool haveGlobalConstants() const {return true;}


    //! @name Methods implementing Operator

    //! Computes the sparse matrix-multivector multiplication.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void apply(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &X, MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &Y, Teuchos::ETransp mode=Teuchos::NO_TRANS, Scalar alpha=ScalarTraits< Scalar >::one(), Scalar beta=ScalarTraits< Scalar >::zero()) const {  }
#else
    void apply(const MultiVector< Scalar, Node > &X, MultiVector< Scalar, Node > &Y, Teuchos::ETransp mode=Teuchos::NO_TRANS, Scalar alpha=ScalarTraits< Scalar >::one(), Scalar beta=ScalarTraits< Scalar >::zero()) const {  }
#endif

    //! Returns the Map associated with the domain of this operator. This will be null until fillComplete() is called.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  getDomainMap() const { return Teuchos::null; }
#else
    const RCP< const Map<Node > >  getDomainMap() const { return Teuchos::null; }
#endif

    //!
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    const RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >  getRangeMap() const { return Teuchos::null; }
#else
    const RCP< const Map<Node > >  getRangeMap() const { return Teuchos::null; }
#endif


    //! @name Overridden from Teuchos::Describable

    //! A simple one-line description of this object.
    std::string description() const { return std::string(""); }

    //! Print the object with some verbosity level to an FancyOStream object.
    void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel=Teuchos::Describable::verbLevel_default) const {  }

    //! Deep copy constructor
    TpetraBlockCrsMatrix(const TpetraBlockCrsMatrix& matrix) {}

    //! Get a copy of the diagonal entries owned by this node, with local row idices 
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void getLocalDiagCopy(Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &diag) const {    }
#else
    void getLocalDiagCopy(Vector< Scalar, Node > &diag) const {    }
#endif

    //! Get offsets of the diagonal entries in the matrix.
    void getLocalDiagOffsets(Teuchos::ArrayRCP<size_t> &offsets) const {    }

    //! Get a copy of the diagonal entries owned by this node, with local row indices.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void getLocalDiagCopy(Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &diag, const Teuchos::ArrayView<const size_t> &offsets) const
#else
    void getLocalDiagCopy(Vector< Scalar, Node > &diag, const Teuchos::ArrayView<const size_t> &offsets) const
#endif
    {}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void replaceDiag(Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &diag) const {    }
#else
    void replaceDiag(Vector< Scalar, Node > &diag) const {    }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void leftScale (const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& x) { }
    void rightScale (const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& x) { }
#else
    void leftScale (const Vector<Scalar, Node>& x) { }
    void rightScale (const Vector<Scalar, Node>& x) { }
#endif

    //! Implements DistObject interface

    //! Access function for the Tpetra::Map this DistObject was constructed with.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > getMap() const { return Teuchos::null; }
#else
    Teuchos::RCP< const Map<Node > > getMap() const { return Teuchos::null; }
#endif

    //! Import.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doImport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &source,
                  const Import< LocalOrdinal, GlobalOrdinal, Node > &importer, CombineMode CM)
#else
    void doImport(const DistObject<char,Node> &source,
                  const Import<Node > &importer, CombineMode CM)
#endif
    {}

    //! Export.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doExport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &dest,
                  const Import< LocalOrdinal, GlobalOrdinal, Node >& importer, CombineMode CM)
#else
    void doExport(const DistObject<char,Node> &dest,
                  const Import<Node >& importer, CombineMode CM)
#endif
    {}

    //! Import (using an Exporter).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doImport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &source,
                  const Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM)
#else
    void doImport(const DistObject<char,Node> &source,
                  const Export<Node >& exporter, CombineMode CM)
#endif
    {}

    //! Export (using an Importer).
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void doExport(const DistObject<char, LocalOrdinal, GlobalOrdinal, Node> &dest,
                  const Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM)
#else
    void doExport(const DistObject<char,Node> &dest,
                  const Export<Node >& exporter, CombineMode CM)
#endif
    {}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void removeEmptyProcessesInPlace (const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> >& newMap)
#else
    void removeEmptyProcessesInPlace (const Teuchos::RCP<const Map<Node> >& newMap)
#endif
    {}



    //! @name Xpetra specific

    //! Does this have an underlying matrix
    bool hasMatrix() const { return false; }

    //! TpetraBlockCrsMatrix constructor to wrap a Tpetra::BlockCrsMatrix object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraBlockCrsMatrix(const Teuchos::RCP<Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &mtx) {
#else
    TpetraBlockCrsMatrix(const Teuchos::RCP<Tpetra::BlockCrsMatrix<Scalar, Node> > &mtx) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

    //! Get the underlying Tpetra matrix
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<const Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > getTpetra_BlockCrsMatrix() const { return Teuchos::null; }
#else
    RCP<const Tpetra::BlockCrsMatrix<Scalar, Node> > getTpetra_BlockCrsMatrix() const { return Teuchos::null; }
#endif

    //! Get the underlying Tpetra matrix
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > getTpetra_BlockCrsMatrixNonConst() const { return Teuchos::null; }
#else
    RCP<Tpetra::BlockCrsMatrix<Scalar, Node> > getTpetra_BlockCrsMatrixNonConst() const { return Teuchos::null; }
#endif

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef HAVE_XPETRA_TPETRA
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef typename Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type local_matrix_type;
#else
    typedef typename Xpetra::CrsMatrix<Scalar, Node>::local_matrix_type local_matrix_type;
#endif

    local_matrix_type getLocalMatrix () const {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix does not support getLocalMatrix due to missing Kokkos::CrsMatrix in Tpetra's experimental implementation");
      local_matrix_type ret;
      TEUCHOS_UNREACHABLE_RETURN(ret);
    }

    void setAllValues (const typename local_matrix_type::row_map_type& ptr,
                       const typename local_matrix_type::StaticCrsGraphType::entries_type::non_const_type& ind,
                       const typename local_matrix_type::values_type& val)
    {
      throw std::runtime_error("Xpetra::TpetraBlockCrsMatrix does not support setAllValues due to missing Kokkos::CrsMatrix in Tpetra's experimental implementation");
    }
#endif  // HAVE_XPETRA_TPETRA
#endif  // HAVE_XPETRA_KOKKOS_REFACTOR

    }; // TpetraBlockCrsMatrix class


#endif  // IF ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_LONG_LONG))) 




#endif // HAVE_XPETRA_EPETRA


} // Xpetra namespace

#endif // XPETRA_TPETRABLOCKCRSMATRIX_DEF_HPP


