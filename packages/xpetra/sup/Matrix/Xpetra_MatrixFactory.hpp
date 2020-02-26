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

#ifndef XPETRA_MATRIXFACTORY_HPP
#define XPETRA_MATRIXFACTORY_HPP

#include "Xpetra_ConfigDefs.hpp"
#include "Xpetra_MapExtractor_fwd.hpp"
#include "Xpetra_Matrix.hpp"
#include "Xpetra_CrsMatrixWrap.hpp"
#include "Xpetra_BlockedCrsMatrix_fwd.hpp"
#include "Xpetra_Map.hpp"
#include "Xpetra_Vector.hpp"
#include "Xpetra_Exceptions.hpp"

namespace Xpetra {

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node = KokkosClassic::DefaultNode::DefaultNodeType>
#else
  template <class Scalar, class Node = KokkosClassic::DefaultNode::DefaultNodeType>
#endif
  class MatrixFactory2 {
#undef XPETRA_MATRIXFACTORY2_SHORT
#include "Xpetra_UseShortNames.hpp"

  public:
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > BuildCopy(const RCP<const Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > A) {
#else
    using LocalOrdinal = typename Tpetra::Map<>::local_ordinal_type;
    using GlobalOrdinal = typename Tpetra::Map<>::global_ordinal_type;
    static RCP<Xpetra::Matrix<Scalar, Node> > BuildCopy(const RCP<const Xpetra::Matrix<Scalar, Node> > A) {
#endif
      RCP<const CrsMatrixWrap> oldOp = Teuchos::rcp_dynamic_cast<const CrsMatrixWrap>(A);
      if (oldOp == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");

      RCP<const CrsMatrix> oldCrsOp = oldOp->getCrsMatrix();

      UnderlyingLib lib = A->getRowMap()->lib();

      TEUCHOS_TEST_FOR_EXCEPTION(lib != UseEpetra && lib != UseTpetra, Exceptions::RuntimeError,
                                 "Not Epetra or Tpetra matrix");

#ifdef HAVE_XPETRA_EPETRA
      if (lib == UseEpetra) {
        // NOTE: The proper Epetra conversion in Xpetra_MatrixFactory.cpp
        throw Exceptions::RuntimeError("Xpetra::BuildCopy(): matrix templates are incompatible with Epetra");
      }
#endif

#ifdef HAVE_XPETRA_TPETRA
      if (lib == UseTpetra) {
        // Underlying matrix is Tpetra
        RCP<const TpetraCrsMatrix> oldTCrsOp = Teuchos::rcp_dynamic_cast<const TpetraCrsMatrix>(oldCrsOp);

        if (oldTCrsOp != Teuchos::null) {
          RCP<TpetraCrsMatrix> newTCrsOp(new TpetraCrsMatrix(*oldTCrsOp));
          RCP<CrsMatrixWrap>   newOp    (new CrsMatrixWrap(Teuchos::as<RCP<CrsMatrix> >(newTCrsOp)));
          newOp->SetFixedBlockSize(A->GetFixedBlockSize());

          return newOp;
        } else {
          throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::TpetraCrsMatrix failed");
        }
      }
#endif

      return Teuchos::null;
    }
  };
#define XPETRA_MATRIXFACTORY2_SHORT

  //template<>
  //class MatrixFactory2<double,int,int,typename Xpetra::Matrix<double, int, int>::node_type> {
  template<class Node>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  class MatrixFactory2<double,int,int,Node> {
#else
  class MatrixFactory2<double,Node> {
#endif
    typedef double                                        Scalar;
    typedef int                                           LocalOrdinal;
    typedef int                                           GlobalOrdinal;
    //typedef Matrix<double, int, GlobalOrdinal>::node_type Node;
#undef XPETRA_MATRIXFACTORY2_SHORT
#include "Xpetra_UseShortNames.hpp"
  public:
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > BuildCopy(const RCP<const Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > A) {
#else
    static RCP<Xpetra::Matrix<Scalar, Node> > BuildCopy(const RCP<const Xpetra::Matrix<Scalar, Node> > A) {
#endif
      RCP<const CrsMatrixWrap> oldOp = Teuchos::rcp_dynamic_cast<const CrsMatrixWrap>(A);
      if (oldOp == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");

      RCP<const CrsMatrix> oldCrsOp = oldOp->getCrsMatrix();

#ifdef HAVE_XPETRA_EPETRA
#ifndef XPETRA_EPETRA_NO_32BIT_GLOBAL_INDICES
      RCP<const EpetraCrsMatrixT<GlobalOrdinal,Node> > oldECrsOp = Teuchos::rcp_dynamic_cast<const EpetraCrsMatrixT<GlobalOrdinal,Node> >(oldCrsOp);
      if (oldECrsOp != Teuchos::null) {
        // Underlying matrix is Epetra
        RCP<CrsMatrix>     newECrsOp(new EpetraCrsMatrixT<GlobalOrdinal,Node>(*oldECrsOp));
        RCP<CrsMatrixWrap> newOp    (new CrsMatrixWrap  (newECrsOp));
        newOp->SetFixedBlockSize(A->GetFixedBlockSize());
        return newOp;
      }
#endif
#endif

#ifdef HAVE_XPETRA_TPETRA
      // Underlying matrix is Tpetra
      RCP<const TpetraCrsMatrix> oldTCrsOp = Teuchos::rcp_dynamic_cast<const TpetraCrsMatrix>(oldCrsOp);
      if (oldTCrsOp != Teuchos::null) {
        RCP<CrsMatrix>     newTCrsOp(new TpetraCrsMatrix(*oldTCrsOp));
        RCP<CrsMatrixWrap> newOp    (new CrsMatrixWrap(newTCrsOp));
        newOp->SetFixedBlockSize(A->GetFixedBlockSize());
        return newOp;
      }
      return Teuchos::null;
#else
      throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::EpetraCrsMatrix or Xpetra::TpetraCrsMatrix failed");
      TEUCHOS_UNREACHABLE_RETURN(Teuchos::null); // make compiler happy
#endif

    } //BuildCopy
  };

#define XPETRA_MATRIXFACTORY2_SHORT

#ifdef HAVE_XPETRA_INT_LONG_LONG
  //template<>
  //class MatrixFactory2<double,int,long long,typename Xpetra::Matrix<double, int, long long>::node_type> {
  template<class Node>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  class MatrixFactory2<double, int, long long, Node> {
#else
  class MatrixFactory2<double, Node> {
#endif
    typedef double                                        Scalar;
    typedef int                                           LocalOrdinal;
    typedef long long                                     GlobalOrdinal;
    //typedef Matrix<double, int, GlobalOrdinal>::node_type Node;
#undef XPETRA_MATRIXFACTORY2_SHORT
#include "Xpetra_UseShortNames.hpp"
  public:
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > BuildCopy(const RCP<const Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > A) {
#else
    static RCP<Xpetra::Matrix<Scalar, Node> > BuildCopy(const RCP<const Xpetra::Matrix<Scalar, Node> > A) {
#endif
      RCP<const CrsMatrixWrap> oldOp = Teuchos::rcp_dynamic_cast<const CrsMatrixWrap>(A);
      if (oldOp == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");

      RCP<const CrsMatrix> oldCrsOp = oldOp->getCrsMatrix();

#ifdef HAVE_XPETRA_EPETRA
#ifndef XPETRA_EPETRA_NO_64BIT_GLOBAL_INDICES
          RCP<const EpetraCrsMatrixT<GlobalOrdinal,Node> > oldECrsOp = Teuchos::rcp_dynamic_cast<const EpetraCrsMatrixT<GlobalOrdinal,Node> >(oldCrsOp);
      if (oldECrsOp != Teuchos::null) {
        // Underlying matrix is Epetra
        RCP<CrsMatrix>     newECrsOp(new EpetraCrsMatrixT<GlobalOrdinal,Node>(*oldECrsOp));
        RCP<CrsMatrixWrap> newOp    (new CrsMatrixWrap  (newECrsOp));
        newOp->SetFixedBlockSize(A->GetFixedBlockSize());
        return newOp;
      }
#endif
#endif

#ifdef HAVE_XPETRA_TPETRA
      // Underlying matrix is Tpetra
      RCP<const TpetraCrsMatrix> oldTCrsOp = Teuchos::rcp_dynamic_cast<const TpetraCrsMatrix>(oldCrsOp);
      if (oldTCrsOp != Teuchos::null) {
        RCP<CrsMatrix>     newTCrsOp(new TpetraCrsMatrix(*oldTCrsOp));
        RCP<CrsMatrixWrap> newOp    (new CrsMatrixWrap(newTCrsOp));
        newOp->SetFixedBlockSize(A->GetFixedBlockSize());
        return newOp;
      }
#else
      throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::EpetraCrsMatrix or Xpetra::TpetraCrsMatrix failed");
#endif

      return Teuchos::null;  // make compiler happy
    }
  };
#endif // HAVE_XPETRA_INT_LONG_LONG

#define XPETRA_MATRIXFACTORY2_SHORT


  template <class Scalar,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            class LocalOrdinal,
            class GlobalOrdinal,
#endif
            class Node>
  class MatrixFactory {
#undef XPETRA_MATRIXFACTORY_SHORT
#include "Xpetra_UseShortNames.hpp"

  private:
#ifndef TPETRA_ENABLE_TEMPLATE_ORDINALS
    using LocalOrdinal = typename Tpetra::Map<>::local_ordinal_type;
    using GlobalOrdinal = typename Tpetra::Map<>::global_ordinal_type;
#endif
    //! Private constructor. This is a static class.
    MatrixFactory() {}

  public:
    /// Constructor for an empty, DynamicProfile matrix.
    /// Supports Epetra only, as DynamicProfile no longer exists in Tpetra.
    static RCP<Matrix> Build(const RCP<const Map>& rowMap) {
      return rcp(new CrsMatrixWrap(rowMap));
    }

    //! Constructor specifying the number of non-zeros for all rows.
    static RCP<Matrix> Build(const RCP<const Map>& rowMap, size_t maxNumEntriesPerRow) {
      return rcp(new CrsMatrixWrap(rowMap, maxNumEntriesPerRow));
    }

    //! Constructor specifying the max number of non-zeros per row and providing column map
    static RCP<Matrix> Build(const RCP<const Map>& rowMap, const RCP<const Map>& colMap, size_t maxNumEntriesPerRow) {
      return rcp(new CrsMatrixWrap(rowMap, colMap, maxNumEntriesPerRow));
    }

    //! Constructor specifying the (possibly different) number of entries per row and providing column map
    static RCP<Matrix> Build(const RCP<const Map>& rowMap, const RCP<const Map>& colMap, const ArrayRCP<const size_t> &NumEntriesPerRowToAlloc) {
      return rcp(new CrsMatrixWrap(rowMap, colMap, NumEntriesPerRowToAlloc));
    }

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
    //! Constructor providing a local Kokkos::CrsMatrix together with a row and column map
    static RCP<Matrix> Build (
        const Teuchos::RCP<const Map>& rowMap,
        const Teuchos::RCP<const Map>& colMap,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        const typename Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type& lclMatrix,
#else
        const typename Xpetra::CrsMatrix<Scalar, Node>::local_matrix_type& lclMatrix,
#endif
        const Teuchos::RCP<Teuchos::ParameterList>& params = null)  {
      XPETRA_MONITOR("MatrixFactory::Build");
      return rcp(new CrsMatrixWrap(rowMap, colMap, lclMatrix, params));
    }
    //! Constructor providing a local Kokkos::CrsMatrix together with all maps
    static RCP<Matrix> Build (
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        const typename Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type& lclMatrix,
#else
        const typename Xpetra::CrsMatrix<Scalar, Node>::local_matrix_type& lclMatrix,
#endif
        const Teuchos::RCP<const Map>& rowMap,
        const Teuchos::RCP<const Map>& colMap,
        const Teuchos::RCP<const Map>& domainMap = Teuchos::null,
        const Teuchos::RCP<const Map>& rangeMap = Teuchos::null,
        const Teuchos::RCP<Teuchos::ParameterList>& params = null)  {
      XPETRA_MONITOR("MatrixFactory::Build");
      return rcp(new CrsMatrixWrap(lclMatrix, rowMap, colMap, domainMap, rangeMap, params));
    }
#endif

    //! Constructor specifying (possibly different) number of entries in each row.
    static RCP<Matrix> Build(const RCP<const Map> &rowMap, const ArrayRCP<const size_t> &NumEntriesPerRowToAlloc) {
      return rcp( new CrsMatrixWrap(rowMap, NumEntriesPerRowToAlloc) );
    }

    //! Constructor specifying graph
    static RCP<Matrix> Build(const RCP<const CrsGraph>& graph, const RCP<ParameterList>& paramList = Teuchos::null) {
      return rcp(new CrsMatrixWrap(graph, paramList));
    }

    //! Constructor for creating a diagonal Xpetra::Matrix using the entries of a given vector for the diagonal
    static RCP<Matrix> Build(const RCP<const Vector>& diagonal) {
      Teuchos::ArrayRCP<const Scalar>         vals             = diagonal->getData(0);
      LocalOrdinal                            NumMyElements    = diagonal->getMap()->getNodeNumElements();
      Teuchos::ArrayView<const GlobalOrdinal> MyGlobalElements = diagonal->getMap()->getNodeElementList();

      Teuchos::RCP<CrsMatrixWrap> mtx = Teuchos::rcp(new CrsMatrixWrap(diagonal->getMap(), 1));

      for (LocalOrdinal i = 0; i < NumMyElements; ++i) {
          mtx->insertGlobalValues(MyGlobalElements[i],
                                  Teuchos::tuple<GlobalOrdinal>(MyGlobalElements[i]),
                                  Teuchos::tuple<Scalar>(vals[i]) );
      }
      mtx->fillComplete();
      return mtx;
    }

    //! Constructor to create a Matrix using a fusedImport-style construction.  The originalMatrix must be a Xpetra::CrsMatrixWrap under the hood or this will fail.
    static RCP<Matrix> Build(const RCP<const Matrix>& sourceMatrix, const Import& importer, const RCP<const Map>& domainMap = Teuchos::null, const RCP<const Map>& rangeMap = Teuchos::null, const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null) {
      RCP<const CrsMatrixWrap> crsOp = Teuchos::rcp_dynamic_cast<const CrsMatrixWrap>(sourceMatrix);
      if (crsOp == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");

      RCP<CrsMatrix> originalCrs = crsOp->getCrsMatrix();
      RCP<CrsMatrix> newCrs      = CrsMatrixFactory::Build(originalCrs, importer, domainMap, rangeMap, params);
      if (newCrs->hasMatrix())
        return rcp(new CrsMatrixWrap(newCrs));
      else
        return Teuchos::null;
    }

    //! Constructor to create a Matrix using a fusedExport-style construction.  The originalMatrix must be a Xpetra::CrsMatrixWrap under the hood or this will fail.
    static RCP<Matrix> Build(const RCP<const Matrix> & sourceMatrix, const Export &exporter, const RCP<const Map> & domainMap, const RCP<const Map> & rangeMap,const Teuchos::RCP<Teuchos::ParameterList>& params) {
      RCP<const CrsMatrixWrap> crsOp = Teuchos::rcp_dynamic_cast<const CrsMatrixWrap>(sourceMatrix);
      if (crsOp == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");

      RCP<CrsMatrix> originalCrs = crsOp->getCrsMatrix();
      return rcp(new CrsMatrixWrap(CrsMatrixFactory::Build(originalCrs, exporter, domainMap, rangeMap, params)));
    }

    //! Constructor to create a Matrix using a fusedImport-style construction.  The originalMatrix must be a Xpetra::CrsMatrixWrap under the hood or this will fail.
    static RCP<Matrix> Build(const RCP<const Matrix>& sourceMatrix, const Import& RowImporter, const Import& DomainImporter, const RCP<const Map>& domainMap, const RCP<const Map>& rangeMap, const Teuchos::RCP<Teuchos::ParameterList>& params) {
      RCP<const CrsMatrixWrap> crsOp = Teuchos::rcp_dynamic_cast<const CrsMatrixWrap>(sourceMatrix);
      if (crsOp == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");

      RCP<CrsMatrix> originalCrs = crsOp->getCrsMatrix();
      RCP<CrsMatrix> newCrs      = CrsMatrixFactory::Build(originalCrs, RowImporter, Teuchos::rcpFromRef(DomainImporter), domainMap, rangeMap, params);
      if (newCrs->hasMatrix())
        return rcp(new CrsMatrixWrap(newCrs));
      else
        return Teuchos::null;
    }

    //! Constructor to create a Matrix using a fusedExport-style construction.  The originalMatrix must be a Xpetra::CrsMatrixWrap under the hood or this will fail.
    static RCP<Matrix> Build(const RCP<const Matrix> & sourceMatrix, const Export &RowExporter, const Export &DomainExporter, const RCP<const Map> & domainMap = Teuchos::null, const RCP<const Map> & rangeMap = Teuchos::null,const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null) {
      RCP<const CrsMatrixWrap> crsOp = Teuchos::rcp_dynamic_cast<const CrsMatrixWrap>(sourceMatrix);
      if (crsOp == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");

      RCP<CrsMatrix> originalCrs = crsOp->getCrsMatrix();
      RCP<CrsMatrix> newCrs      = CrsMatrixFactory::Build(originalCrs, RowExporter, Teuchos::rcpFromRef(DomainExporter), domainMap, rangeMap, params);
      if (newCrs->hasMatrix())
        return rcp(new CrsMatrixWrap(newCrs));
      else
        return Teuchos::null;
    }


    //! create an explicit copy of a given matrix
    //! This routine supports blocked and single-block operators
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > BuildCopy(const RCP<const Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > A) {
#else
    static RCP<Xpetra::Matrix<Scalar, Node> > BuildCopy(const RCP<const Xpetra::Matrix<Scalar, Node> > A) {
#endif
      RCP<const BlockedCrsMatrix> input = Teuchos::rcp_dynamic_cast<const BlockedCrsMatrix>(A);
      if(input == Teuchos::null)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return Xpetra::MatrixFactory2<Scalar,LocalOrdinal,GlobalOrdinal,Node>::BuildCopy(A);
#else
        return Xpetra::MatrixFactory2<Scalar,Node>::BuildCopy(A);
#endif

      // deep copy of MapExtractors (and underlying maps)
      RCP<const MapExtractor> rgMapExt = Teuchos::rcp(new MapExtractor(*(input->getRangeMapExtractor())));
      RCP<const MapExtractor> doMapExt = Teuchos::rcp(new MapExtractor(*(input->getDomainMapExtractor())));

      // create new BlockedCrsMatrix object
      RCP<BlockedCrsMatrix> bop = Teuchos::rcp(new BlockedCrsMatrix(rgMapExt, doMapExt, input->getNodeMaxNumRowEntries()));

      for (size_t r = 0; r < input->Rows(); ++r) {
        for (size_t c = 0; c < input->Cols(); ++c)
          if(input->getMatrix(r,c) != Teuchos::null) {
            // make a deep copy of the matrix
            // This is a recursive call to this function
            RCP<Matrix> mat =
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                Xpetra::MatrixFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::BuildCopy(input->getMatrix(r,c));
#else
                Xpetra::MatrixFactory<Scalar,Node>::BuildCopy(input->getMatrix(r,c));
#endif
            bop->setMatrix(r,c,mat);
          }
      }

      if(input->isFillComplete())
        bop->fillComplete();
      return bop;
    }
  };
#define XPETRA_MATRIXFACTORY_SHORT

}

#define XPETRA_MATRIXFACTORY_SHORT
#define XPETRA_MATRIXFACTORY2_SHORT
#endif
