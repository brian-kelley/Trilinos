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
#ifndef XPETRA_CRSMATRIXFACTORY_HPP
#define XPETRA_CRSMATRIXFACTORY_HPP

#include "Xpetra_ConfigDefs.hpp"

#include "Xpetra_CrsMatrix.hpp"

#ifdef HAVE_XPETRA_TPETRA
#include "Xpetra_TpetraCrsMatrix.hpp"
#endif

#ifdef HAVE_XPETRA_EPETRA
#include "Xpetra_EpetraCrsMatrix.hpp"
#endif

#include "Xpetra_Exceptions.hpp"

namespace Xpetra {

  template <class Scalar,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            class LocalOrdinal,
            class GlobalOrdinal,
#endif
            class Node = KokkosClassic::DefaultNode::DefaultNodeType>
  class CrsMatrixFactory {
  private:
#ifndef TPETRA_ENABLE_TEMPLATE_ORDINALS
    using LocalOrdinal = typename Tpetra::Map<>::local_ordinal_type;
    using GlobalOrdinal = typename Tpetra::Map<>::global_ordinal_type;
#endif
    //! Private constructor. This is a static class.
    CrsMatrixFactory() {}

  public:
    //! Constructor for empty matrix (intended use is an import/export target - can't insert entries directly)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
    Build (const RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &rowMap)
#else
    static RCP<CrsMatrix<Scalar, Node> >
    Build (const RCP<const Map<Node> > &rowMap)
#endif
    {
      TEUCHOS_TEST_FOR_EXCEPTION(rowMap->lib() == UseEpetra, std::logic_error,
          "Can't create Xpetra::EpetraCrsMatrix with these scalar/LO/GO types");
#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, 0) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, 0) );
#endif
#endif

      XPETRA_FACTORY_END;
    }

    //! Constructor specifying fixed number of entries for each row.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
    Build (const RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &rowMap,
#else
    static RCP<CrsMatrix<Scalar, Node> >
    Build (const RCP<const Map<Node> > &rowMap,
#endif
           size_t maxNumEntriesPerRow,
           const Teuchos::RCP<Teuchos::ParameterList>& plist = Teuchos::null)
    {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, maxNumEntriesPerRow, plist) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, maxNumEntriesPerRow, plist) );
#endif
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(rowMap->lib());
      XPETRA_FACTORY_END;
    }

    //! Constructor specifying (possibly different) number of entries in each row.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
    Build (const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > >& rowMap,
#else
    static RCP<CrsMatrix<Scalar, Node> >
    Build (const Teuchos::RCP< const Map<Node > >& rowMap,
#endif
           const ArrayRCP<const size_t>& NumEntriesPerRowToAlloc,
           const Teuchos::RCP<Teuchos::ParameterList>& plist = Teuchos::null)
    {
#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, NumEntriesPerRowToAlloc, plist) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, NumEntriesPerRowToAlloc, plist) );
#endif
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(rowMap->lib());
      XPETRA_FACTORY_END;
    }

    //! Constructor specifying column Map and fixed number of entries for each row.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
    Build (const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> >& rowMap,
           const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> >& colMap,
#else
    static RCP<CrsMatrix<Scalar, Node> >
    Build (const Teuchos::RCP<const Map<Node> >& rowMap,
           const Teuchos::RCP<const Map<Node> >& colMap,
#endif
           size_t maxNumEntriesPerRow,
           const Teuchos::RCP<Teuchos::ParameterList>& plist = Teuchos::null)
    {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, colMap, maxNumEntriesPerRow, plist) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, colMap, maxNumEntriesPerRow, plist) );
#endif
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(rowMap->lib());
      XPETRA_FACTORY_END;
    }

    //! Constructor specifying column Map and number of entries in each row.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(const Teuchos::RCP< const Map<Node > > &rowMap, const Teuchos::RCP< const Map<Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#endif
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, colMap, NumEntriesPerRowToAlloc, plist) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, colMap, NumEntriesPerRowToAlloc, plist) );
#endif
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(rowMap->lib());
      XPETRA_FACTORY_END;
    }

    //! Constructor specifying a previously constructed graph.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(const Teuchos::RCP< const CrsGraph< LocalOrdinal, GlobalOrdinal, Node > > &graph, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(const Teuchos::RCP< const CrsGraph<Node > > &graph, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#endif
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (graph->getRowMap()->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(graph, plist) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(graph, plist) );
#endif
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(graph->getRowMap()->lib());
      XPETRA_FACTORY_END;
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, LocalOrdinal, GlobalOrdinal, Node > > &sourceMatrix,
        const Import<LocalOrdinal,GlobalOrdinal,Node> &importer,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & domainMap = Teuchos::null,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & rangeMap = Teuchos::null,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, Node > > &sourceMatrix,
        const Import<Node> &importer,
        const RCP<const Map<Node> > & domainMap = Teuchos::null,
        const RCP<const Map<Node> > & rangeMap = Teuchos::null,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null) {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (sourceMatrix->getRowMap()->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(sourceMatrix,importer,domainMap,rangeMap,params));
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(sourceMatrix,importer,domainMap,rangeMap,params));
#endif
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(sourceMatrix->getRowMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, LocalOrdinal, GlobalOrdinal, Node > > &sourceMatrix,
        const Export<LocalOrdinal,GlobalOrdinal,Node> &exporter,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, Node > > &sourceMatrix,
        const Export<Node> &exporter,
#endif
        const RCP<Map<LocalOrdinal,GlobalOrdinal,Scalar> > & domainMap = Teuchos::null,
        const RCP<Map<LocalOrdinal,GlobalOrdinal,Scalar> > & rangeMap = Teuchos::null,
        const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null) {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (sourceMatrix->getRowMap()->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(sourceMatrix,exporter,domainMap,rangeMap,params));
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(sourceMatrix,exporter,domainMap,rangeMap,params));
#endif
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(sourceMatrix->getRowMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, LocalOrdinal, GlobalOrdinal, Node > > &sourceMatrix,
        const Import<LocalOrdinal,GlobalOrdinal,Node> &RowImporter,
        const RCP<const Import<LocalOrdinal,GlobalOrdinal,Node> > DomainImporter,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & domainMap,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & rangeMap,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, Node > > &sourceMatrix,
        const Import<Node> &RowImporter,
        const RCP<const Import<Node> > DomainImporter,
        const RCP<const Map<Node> > & domainMap,
        const RCP<const Map<Node> > & rangeMap,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params) {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (sourceMatrix->getRowMap()->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(sourceMatrix,RowImporter,DomainImporter,domainMap,rangeMap,params));
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(sourceMatrix,RowImporter,DomainImporter,domainMap,rangeMap,params));
#endif
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(sourceMatrix->getRowMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, LocalOrdinal, GlobalOrdinal, Node > > &sourceMatrix,
        const Export<LocalOrdinal,GlobalOrdinal,Node> &RowExporter,
        const RCP<const Export<LocalOrdinal,GlobalOrdinal,Node> > DomainExporter,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, Node > > &sourceMatrix,
        const Export<Node> &RowExporter,
        const RCP<const Export<Node> > DomainExporter,
#endif
        const RCP<Map<LocalOrdinal,GlobalOrdinal,Scalar> > & domainMap,
        const RCP<Map<LocalOrdinal,GlobalOrdinal,Scalar> > & rangeMap,
        const Teuchos::RCP<Teuchos::ParameterList>& params) {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (sourceMatrix->getRowMap()->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(sourceMatrix,RowExporter,DomainExporter,domainMap,rangeMap,params));
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(sourceMatrix,RowExporter,DomainExporter,domainMap,rangeMap,params));
#endif
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(sourceMatrix->getRowMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build (
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rowMap,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& colMap,
        const typename Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type& lclMatrix,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build (
        const Teuchos::RCP<const Map<Node> >& rowMap,
        const Teuchos::RCP<const Map<Node> >& colMap,
        const typename Xpetra::CrsMatrix<Scalar, Node>::local_matrix_type& lclMatrix,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params = null)  {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, colMap, lclMatrix, params));
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, colMap, lclMatrix, params));
#endif
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(rowMap->lib());
      XPETRA_FACTORY_END;
    }
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build (
        const typename Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type& lclMatrix,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rowMap,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& colMap,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap = Teuchos::null,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap = Teuchos::null,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build (
        const typename Xpetra::CrsMatrix<Scalar, Node>::local_matrix_type& lclMatrix,
        const Teuchos::RCP<const Map<Node> >& rowMap,
        const Teuchos::RCP<const Map<Node> >& colMap,
        const Teuchos::RCP<const Map<Node> >& domainMap = Teuchos::null,
        const Teuchos::RCP<const Map<Node> >& rangeMap = Teuchos::null,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params = null)  {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(lclMatrix, rowMap, colMap, domainMap, rangeMap, params));
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(lclMatrix, rowMap, colMap, domainMap, rangeMap, params));
#endif
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(rowMap->lib());
      XPETRA_FACTORY_END;
    }
#endif

  };

// we need the Epetra specialization only if Epetra is enabled
#if (defined(HAVE_XPETRA_EPETRA) && !defined(XPETRA_EPETRA_NO_32BIT_GLOBAL_INDICES))

  // Specializtion for SC=double, LO=int, GO=int and Node=EpetraNode
  // Used both for Epetra and Tpetra
  template <>
  class CrsMatrixFactory<double, int, int, EpetraNode> {
    typedef double Scalar;
    typedef int LocalOrdinal;
    typedef int GlobalOrdinal;
    typedef EpetraNode Node;

  private:
    //! Private constructor. This is a static class.
    CrsMatrixFactory() {}

  public:
    //! Constructor for empty matrix (intended use is an import/export target - can't insert entries directly)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
    Build (const RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &rowMap)
#else
    static RCP<CrsMatrix<Scalar, Node> >
    Build (const RCP<const Map<Node> > &rowMap)
#endif
    {
      XPETRA_MONITOR("CrsMatrixFactory::Build");
#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, 0) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, 0) );
#endif
#endif
#ifdef HAVE_XPETRA_EPETRA
      if(rowMap->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<int,Node>(rowMap));
#endif
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(const RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &rowMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(const RCP<const Map<Node> > &rowMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#endif
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, maxNumEntriesPerRow, plist) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, maxNumEntriesPerRow, plist) );
#endif
#endif

      if (rowMap->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<int,Node>(rowMap, maxNumEntriesPerRow, plist) );

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(const Teuchos::RCP< const Map<Node > > &rowMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#endif
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, NumEntriesPerRowToAlloc, plist) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, NumEntriesPerRowToAlloc, plist) );
#endif
#endif

      if (rowMap->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<int,Node>(rowMap, NumEntriesPerRowToAlloc, plist) );

      XPETRA_FACTORY_END;
    }

    //! Constructor specifying column Map and fixed number of entries for each row.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(const Teuchos::RCP< const Map<Node > > &rowMap, const Teuchos::RCP< const Map<Node > > &colMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#endif
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, colMap, maxNumEntriesPerRow, plist) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, colMap, maxNumEntriesPerRow, plist) );
#endif
#endif

      if (rowMap->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<int,Node>(rowMap, colMap, maxNumEntriesPerRow, plist) );

      XPETRA_FACTORY_END;
    }

    //! Constructor specifying column Map and number of entries in each row.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(const Teuchos::RCP< const Map<Node > > &rowMap, const Teuchos::RCP< const Map<Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#endif
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, colMap, NumEntriesPerRowToAlloc, plist) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, colMap, NumEntriesPerRowToAlloc, plist) );
#endif
#endif

      if (rowMap->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<int,Node>(rowMap, colMap, NumEntriesPerRowToAlloc, plist) );

      XPETRA_FACTORY_END;
    }

    //! Constructor specifying a previously constructed graph.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(const Teuchos::RCP< const CrsGraph< LocalOrdinal, GlobalOrdinal, Node > > &graph, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(const Teuchos::RCP< const CrsGraph<Node > > &graph, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#endif
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (graph->getRowMap()->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(graph, plist) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(graph, plist) );
#endif
#endif

      if (graph->getRowMap()->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<int,Node>(graph, plist) );

      XPETRA_FACTORY_END;
    }


    //! Constructor using FusedImport
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, LocalOrdinal, GlobalOrdinal, Node > > &sourceMatrix,
        const Import<LocalOrdinal,GlobalOrdinal,Node> &importer,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & domainMap = Teuchos::null,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & rangeMap = Teuchos::null,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, Node > > &sourceMatrix,
        const Import<Node> &importer,
        const RCP<const Map<Node> > & domainMap = Teuchos::null,
        const RCP<const Map<Node> > & rangeMap = Teuchos::null,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null) {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (sourceMatrix->getRowMap()->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(sourceMatrix,importer,domainMap,rangeMap,params) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(sourceMatrix,importer,domainMap,rangeMap,params) );
#endif
#endif

      if (sourceMatrix->getRowMap()->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<int,Node>(sourceMatrix,importer,domainMap,rangeMap,params) );

      XPETRA_FACTORY_END;
    }

    //! Constructor using FusedExport
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, LocalOrdinal, GlobalOrdinal, Node > > &sourceMatrix,
        const Export<LocalOrdinal,GlobalOrdinal,Node> &exporter,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & domainMap = Teuchos::null,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & rangeMap = Teuchos::null,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, Node > > &sourceMatrix,
        const Export<Node> &exporter,
        const RCP<const Map<Node> > & domainMap = Teuchos::null,
        const RCP<const Map<Node> > & rangeMap = Teuchos::null,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null) {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (sourceMatrix->getRowMap()->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(sourceMatrix,exporter,domainMap,rangeMap,params) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(sourceMatrix,exporter,domainMap,rangeMap,params) );
#endif
#endif

      if (sourceMatrix->getRowMap()->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<int,Node>(sourceMatrix,exporter,domainMap,rangeMap,params) );

      XPETRA_FACTORY_END;
    }

    //! Constructor using FusedImport
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, LocalOrdinal, GlobalOrdinal, Node > > &sourceMatrix,
        const Import<LocalOrdinal,GlobalOrdinal,Node> & RowImporter,
        const RCP<const Import<LocalOrdinal,GlobalOrdinal,Node> > DomainImporter,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & domainMap,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & rangeMap,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, Node > > &sourceMatrix,
        const Import<Node> & RowImporter,
        const RCP<const Import<Node> > DomainImporter,
        const RCP<const Map<Node> > & domainMap,
        const RCP<const Map<Node> > & rangeMap,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params) {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (sourceMatrix->getRowMap()->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(sourceMatrix,RowImporter,DomainImporter,domainMap,rangeMap,params) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(sourceMatrix,RowImporter,DomainImporter,domainMap,rangeMap,params) );
#endif
#endif

      if (sourceMatrix->getRowMap()->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<int,Node>(sourceMatrix,RowImporter,DomainImporter,domainMap,rangeMap,params) );

      XPETRA_FACTORY_END;
    }

    //! Constructor using FusedExport
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, LocalOrdinal, GlobalOrdinal, Node > > &sourceMatrix,
        const Export<LocalOrdinal,GlobalOrdinal,Node> &RowExporter,
        const RCP<const Export<LocalOrdinal,GlobalOrdinal,Node> > DomainExporter,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & domainMap,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & rangeMap,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, Node > > &sourceMatrix,
        const Export<Node> &RowExporter,
        const RCP<const Export<Node> > DomainExporter,
        const RCP<const Map<Node> > & domainMap,
        const RCP<const Map<Node> > & rangeMap,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params) {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (sourceMatrix->getRowMap()->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(sourceMatrix,RowExporter,DomainExporter,domainMap,rangeMap,params) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(sourceMatrix,RowExporter,DomainExporter,domainMap,rangeMap,params) );
#endif
#endif

      if (sourceMatrix->getRowMap()->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<int,Node>(sourceMatrix,RowExporter,DomainExporter,domainMap,rangeMap,params) );

      XPETRA_FACTORY_END;
    }


#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build (
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rowMap,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& colMap,
        const typename Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type& lclMatrix,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build (
        const Teuchos::RCP<const Map<Node> >& rowMap,
        const Teuchos::RCP<const Map<Node> >& colMap,
        const typename Xpetra::CrsMatrix<Scalar, Node>::local_matrix_type& lclMatrix,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params = null)  {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, colMap, lclMatrix, params));
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, colMap, lclMatrix, params));
#endif
#endif

      if (rowMap->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<int,Node>(rowMap, colMap, lclMatrix, params) );

      XPETRA_FACTORY_END;
    }
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build (
        const typename Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type& lclMatrix,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rowMap,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& colMap,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap = Teuchos::null,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap = Teuchos::null,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build (
        const typename Xpetra::CrsMatrix<Scalar, Node>::local_matrix_type& lclMatrix,
        const Teuchos::RCP<const Map<Node> >& rowMap,
        const Teuchos::RCP<const Map<Node> >& colMap,
        const Teuchos::RCP<const Map<Node> >& domainMap = Teuchos::null,
        const Teuchos::RCP<const Map<Node> >& rangeMap = Teuchos::null,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params = null)  {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(lclMatrix, rowMap, colMap, domainMap, rangeMap, params));
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(lclMatrix, rowMap, colMap, domainMap, rangeMap, params));
#endif
#endif

      if (rowMap->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<int,Node>(lclMatrix, rowMap, colMap, domainMap, rangeMap, params) );

      XPETRA_FACTORY_END;
    }
#endif

  };
#endif

// we need the Epetra specialization only if Epetra is enabled
#if (defined(HAVE_XPETRA_EPETRA) && !defined(XPETRA_EPETRA_NO_64BIT_GLOBAL_INDICES))

  template <>
  class CrsMatrixFactory<double, int, long long, EpetraNode> {
    typedef double Scalar;
    typedef int LocalOrdinal;
    typedef long long GlobalOrdinal;
    typedef EpetraNode Node;

  private:
    //! Private constructor. This is a static class.
    CrsMatrixFactory() {}

  public:
    //! Constructor for empty matrix (intended use is an import/export target - can't insert entries directly)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
    Build (const RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &rowMap)
#else
    static RCP<CrsMatrix<Scalar, Node> >
    Build (const RCP<const Map<Node> > &rowMap)
#endif
    {
      XPETRA_MONITOR("CrsMatrixFactory::Build");
#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, 0) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, 0) );
#endif
#endif
#ifdef HAVE_XPETRA_EPETRA
      if(rowMap->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<long long,Node>(rowMap));
#endif
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(const RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > &rowMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(const RCP<const Map<Node> > &rowMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#endif
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, maxNumEntriesPerRow, plist) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, maxNumEntriesPerRow, plist) );
#endif
#endif

      if (rowMap->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<long long, Node>(rowMap, maxNumEntriesPerRow, plist) );

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(const Teuchos::RCP< const Map<Node > > &rowMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#endif
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, NumEntriesPerRowToAlloc, plist) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, NumEntriesPerRowToAlloc, plist) );
#endif
#endif

      if (rowMap->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<long long, Node>(rowMap, NumEntriesPerRowToAlloc, plist) );

      XPETRA_FACTORY_END;
    }

    //! Constructor specifying column Map and fixed number of entries for each row.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(const Teuchos::RCP< const Map<Node > > &rowMap, const Teuchos::RCP< const Map<Node > > &colMap, size_t maxNumEntriesPerRow, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#endif
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, colMap, maxNumEntriesPerRow, plist) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, colMap, maxNumEntriesPerRow, plist) );
#endif
#endif

      if (rowMap->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<long long, Node>(rowMap, colMap, maxNumEntriesPerRow, plist) );

      XPETRA_FACTORY_END;
    }

    //! Constructor specifying column Map and number of entries in each row.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &rowMap, const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(const Teuchos::RCP< const Map<Node > > &rowMap, const Teuchos::RCP< const Map<Node > > &colMap, const ArrayRCP< const size_t > &NumEntriesPerRowToAlloc, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#endif
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, colMap, NumEntriesPerRowToAlloc, plist) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, colMap, NumEntriesPerRowToAlloc, plist) );
#endif
#endif

      if (rowMap->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<long long, Node>(rowMap, colMap, NumEntriesPerRowToAlloc, plist) );

      XPETRA_FACTORY_END;
    }

    //! Constructor specifying a previously constructed graph.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(const Teuchos::RCP< const CrsGraph< LocalOrdinal, GlobalOrdinal, Node > > &graph, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(const Teuchos::RCP< const CrsGraph<Node > > &graph, const Teuchos::RCP< Teuchos::ParameterList > &plist=Teuchos::null) {
#endif
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (graph->getRowMap()->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(graph, plist) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(graph, plist) );
#endif
#endif

      if (graph->getRowMap()->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<long long, Node>(graph, plist) );

      XPETRA_FACTORY_END;
    }


    //! Constructor using FusedImport
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, LocalOrdinal, GlobalOrdinal, Node > > &sourceMatrix,
        const Import<LocalOrdinal,GlobalOrdinal,Node> &importer,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & domainMap = Teuchos::null,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & rangeMap = Teuchos::null,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, Node > > &sourceMatrix,
        const Import<Node> &importer,
        const RCP<const Map<Node> > & domainMap = Teuchos::null,
        const RCP<const Map<Node> > & rangeMap = Teuchos::null,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null) {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (sourceMatrix->getRowMap()->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(sourceMatrix,importer,domainMap,rangeMap,params) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(sourceMatrix,importer,domainMap,rangeMap,params) );
#endif
#endif

      if (sourceMatrix->getRowMap()->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<long long, Node>(sourceMatrix,importer,domainMap,rangeMap,params) );

      XPETRA_FACTORY_END;
    }

    //! Constructor using FusedExport
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, LocalOrdinal, GlobalOrdinal, Node > > &sourceMatrix,
        const Export<LocalOrdinal,GlobalOrdinal,Node> &exporter,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & domainMap = Teuchos::null,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & rangeMap = Teuchos::null,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, Node > > &sourceMatrix,
        const Export<Node> &exporter,
        const RCP<const Map<Node> > & domainMap = Teuchos::null,
        const RCP<const Map<Node> > & rangeMap = Teuchos::null,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null) {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (sourceMatrix->getRowMap()->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(sourceMatrix,exporter,domainMap,rangeMap,params) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(sourceMatrix,exporter,domainMap,rangeMap,params) );
#endif
#endif

      if (sourceMatrix->getRowMap()->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<long long, Node>(sourceMatrix,exporter,domainMap,rangeMap,params) );

      XPETRA_FACTORY_END;
    }

    //! Constructor using FusedImport
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, LocalOrdinal, GlobalOrdinal, Node > > &sourceMatrix,
        const Import<LocalOrdinal,GlobalOrdinal,Node> & RowImporter,
        const RCP<const Import<LocalOrdinal,GlobalOrdinal,Node> > DomainImporter,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & domainMap,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & rangeMap,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, Node > > &sourceMatrix,
        const Import<Node> & RowImporter,
        const RCP<const Import<Node> > DomainImporter,
        const RCP<const Map<Node> > & domainMap,
        const RCP<const Map<Node> > & rangeMap,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params) {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (sourceMatrix->getRowMap()->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(sourceMatrix,RowImporter,DomainImporter,domainMap,rangeMap,params) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(sourceMatrix,RowImporter,DomainImporter,domainMap,rangeMap,params) );
#endif
#endif

      if (sourceMatrix->getRowMap()->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<long long,Node>(sourceMatrix,RowImporter,DomainImporter,domainMap,rangeMap,params) );

      XPETRA_FACTORY_END;
    }

    //! Constructor using FusedExport
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, LocalOrdinal, GlobalOrdinal, Node > > &sourceMatrix,
        const Export<LocalOrdinal,GlobalOrdinal,Node> &RowExporter,
        const RCP<const Export<LocalOrdinal,GlobalOrdinal,Node> > DomainExporter,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & domainMap,
        const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > & rangeMap,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build(
        const Teuchos::RCP< const CrsMatrix< Scalar, Node > > &sourceMatrix,
        const Export<Node> &RowExporter,
        const RCP<const Export<Node> > DomainExporter,
        const RCP<const Map<Node> > & domainMap,
        const RCP<const Map<Node> > & rangeMap,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params) {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (sourceMatrix->getRowMap()->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(sourceMatrix,RowExporter,DomainExporter,domainMap,rangeMap,params) );
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(sourceMatrix,RowExporter,DomainExporter,domainMap,rangeMap,params) );
#endif
#endif

      if (sourceMatrix->getRowMap()->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<long long,Node>(sourceMatrix,RowExporter,DomainExporter,domainMap,rangeMap,params) );

      XPETRA_FACTORY_END;
    }

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build (
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rowMap,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& colMap,
        const typename Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type& lclMatrix,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build (
        const Teuchos::RCP<const Map<Node> >& rowMap,
        const Teuchos::RCP<const Map<Node> >& colMap,
        const typename Xpetra::CrsMatrix<Scalar, Node>::local_matrix_type& lclMatrix,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params = null)  {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(rowMap, colMap, lclMatrix, params));
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(rowMap, colMap, lclMatrix, params));
#endif
#endif

      if (rowMap->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<long long, Node>(rowMap, colMap, lclMatrix, params) );

      XPETRA_FACTORY_END;
    }
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Build (
        const typename Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type& lclMatrix,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rowMap,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& colMap,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& domainMap = Teuchos::null,
        const Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& rangeMap = Teuchos::null,
#else
    static RCP<CrsMatrix<Scalar, Node> > Build (
        const typename Xpetra::CrsMatrix<Scalar, Node>::local_matrix_type& lclMatrix,
        const Teuchos::RCP<const Map<Node> >& rowMap,
        const Teuchos::RCP<const Map<Node> >& colMap,
        const Teuchos::RCP<const Map<Node> >& domainMap = Teuchos::null,
        const Teuchos::RCP<const Map<Node> >& rangeMap = Teuchos::null,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params = null)  {
      XPETRA_MONITOR("CrsMatrixFactory::Build");

#ifdef HAVE_XPETRA_TPETRA
      if (rowMap->lib() == UseTpetra)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp( new TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(lclMatrix, rowMap, colMap, domainMap, rangeMap, params));
#else
        return rcp( new TpetraCrsMatrix<Scalar, Node>(lclMatrix, rowMap, colMap, domainMap, rangeMap, params));
#endif
#endif

      if (rowMap->lib() == UseEpetra)
        return rcp( new EpetraCrsMatrixT<long long, Node>(lclMatrix, rowMap, colMap, domainMap, rangeMap, params) );

      XPETRA_FACTORY_END;
    }
#endif

  };
#endif

}

#define XPETRA_CRSMATRIXFACTORY_SHORT
#endif
