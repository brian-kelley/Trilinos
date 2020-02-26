// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
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
#ifndef MUELU_UTILITIES_DECL_HPP
#define MUELU_UTILITIES_DECL_HPP

#include <string>

#include "MueLu_ConfigDefs.hpp"

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_ParameterList.hpp>

#ifdef HAVE_MUELU_TPETRA
#include <Xpetra_TpetraBlockCrsMatrix.hpp>
#endif
#include <Xpetra_BlockedCrsMatrix_fwd.hpp>
#include <Xpetra_CrsMatrix_fwd.hpp>
#include <Xpetra_CrsMatrixWrap_fwd.hpp>
#include <Xpetra_Map_fwd.hpp>
#include <Xpetra_MapFactory_fwd.hpp>
#include <Xpetra_Matrix_fwd.hpp>
#include <Xpetra_MatrixFactory_fwd.hpp>
#include <Xpetra_MultiVector_fwd.hpp>
#include <Xpetra_MultiVectorFactory_fwd.hpp>
#include <Xpetra_Operator_fwd.hpp>
#include <Xpetra_Vector_fwd.hpp>
#include <Xpetra_VectorFactory_fwd.hpp>
#include <Xpetra_ExportFactory.hpp>

#include <Xpetra_Import.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_MatrixMatrix.hpp>

#ifdef HAVE_MUELU_EPETRA
#include <Xpetra_EpetraCrsMatrix_fwd.hpp>

// needed because of inlined function
//TODO: remove inline function?
#include <Xpetra_EpetraCrsMatrix.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>

#endif

#include "MueLu_Exceptions.hpp"

#ifdef HAVE_MUELU_EPETRAEXT
class Epetra_CrsMatrix;
class Epetra_MultiVector;
class Epetra_Vector;
#include "EpetraExt_Transpose_RowMatrix.h"
#endif

#ifdef HAVE_MUELU_TPETRA
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_FECrsMatrix.hpp>
#include <Tpetra_RowMatrixTransposer.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_FEMultiVector.hpp>
#include <Xpetra_TpetraCrsMatrix_fwd.hpp>
#include <Xpetra_TpetraMultiVector_fwd.hpp>
#endif

#include <MueLu_UtilitiesBase.hpp>


namespace MueLu {

#ifdef HAVE_MUELU_EPETRA
  //defined after Utilities class
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<typename SC,typename LO,typename GO,typename NO>
  RCP<Xpetra::CrsMatrixWrap<SC,LO,GO,NO> >
#else
  template<typename SC,typename NO>
  RCP<Xpetra::CrsMatrixWrap<SC,NO> >
#endif
  Convert_Epetra_CrsMatrix_ToXpetra_CrsMatrixWrap(RCP<Epetra_CrsMatrix> &epAB);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<typename SC,typename LO,typename GO,typename NO>
  RCP<Xpetra::Matrix<SC, LO, GO, NO> >
#else
  template<typename SC,typename NO>
  RCP<Xpetra::Matrix<SC, NO> >
#endif
  EpetraCrs_To_XpetraMatrix(const Teuchos::RCP<Epetra_CrsMatrix>& A);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<typename SC,typename LO,typename GO,typename NO>
  RCP<Xpetra::MultiVector<SC, LO, GO, NO> >
#else
  template<typename SC,typename NO>
  RCP<Xpetra::MultiVector<SC, NO> >
#endif
  EpetraMultiVector_To_XpetraMultiVector(const Teuchos::RCP<Epetra_MultiVector>& V);
#endif

#ifdef HAVE_MUELU_TPETRA
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<typename SC,typename LO,typename GO,typename NO>
  RCP<Xpetra::Matrix<SC, LO, GO, NO> >
  TpetraCrs_To_XpetraMatrix(const Teuchos::RCP<Tpetra::CrsMatrix<SC, LO, GO, NO> >& Atpetra);

  template<typename SC,typename LO,typename GO,typename NO>
  RCP<Xpetra::Matrix<SC, LO, GO, NO> >
  TpetraFECrs_To_XpetraMatrix(const Teuchos::RCP<Tpetra::FECrsMatrix<SC, LO, GO, NO> >& Atpetra);

  template<typename SC,typename LO,typename GO,typename NO>
  RCP<Xpetra::MultiVector<SC, LO, GO, NO> >
  TpetraMultiVector_To_XpetraMultiVector(const Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NO> >& Vtpetra);

  template<typename SC,typename LO,typename GO,typename NO>
  RCP<Xpetra::MultiVector<SC, LO, GO, NO> >
  TpetraFEMultiVector_To_XpetraMultiVector(const Teuchos::RCP<Tpetra::FEMultiVector<SC, LO, GO, NO> >& Vtpetra);
#else
  template<typename SC,typename NO>
  RCP<Xpetra::Matrix<SC, NO> >
  TpetraCrs_To_XpetraMatrix(const Teuchos::RCP<Tpetra::CrsMatrix<SC, NO> >& Atpetra);

  template<typename SC,typename NO>
  RCP<Xpetra::Matrix<SC, NO> >
  TpetraFECrs_To_XpetraMatrix(const Teuchos::RCP<Tpetra::FECrsMatrix<SC, NO> >& Atpetra);

  template<typename SC,typename NO>
  RCP<Xpetra::MultiVector<SC, NO> >
  TpetraMultiVector_To_XpetraMultiVector(const Teuchos::RCP<Tpetra::MultiVector<SC, NO> >& Vtpetra);

  template<typename SC,typename NO>
  RCP<Xpetra::MultiVector<SC, NO> >
  TpetraFEMultiVector_To_XpetraMultiVector(const Teuchos::RCP<Tpetra::FEMultiVector<SC, NO> >& Vtpetra);
#endif
#endif

  /*!
    @class Utilities
    @brief MueLu utility class.

    This class provides a number of static helper methods. Some are temporary and will eventually
    go away, while others should be moved to Xpetra.
    */
  template <class Scalar,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            class LocalOrdinal = DefaultLocalOrdinal,
            class GlobalOrdinal = DefaultGlobalOrdinal,
#endif
            class Node = DefaultNode>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  class Utilities : public UtilitiesBase<Scalar, LocalOrdinal, GlobalOrdinal, Node> {
#else
  class Utilities : public UtilitiesBase<Scalar, Node> {
#endif
#undef MUELU_UTILITIES_SHORT
#ifndef TPETRA_ENABLE_TEMPLATE_ORDINALS
    using LocalOrdinal = typename Tpetra::Map<>::local_ordinal_type;
    using GlobalOrdinal = typename Tpetra::Map<>::global_ordinal_type;
#endif
    //#include "MueLu_UseShortNames.hpp"

  public:
    typedef typename Teuchos::ScalarTraits<Scalar>::magnitudeType Magnitude;

#ifdef HAVE_MUELU_EPETRA
    //! Helper utility to pull out the underlying Epetra objects from an Xpetra object
    // @{
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<const Epetra_MultiVector>                    MV2EpetraMV(RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > const vec);
    static RCP<      Epetra_MultiVector>                    MV2NonConstEpetraMV(RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > vec);
#else
    static RCP<const Epetra_MultiVector>                    MV2EpetraMV(RCP<Xpetra::MultiVector<Scalar,Node> > const vec);
    static RCP<      Epetra_MultiVector>                    MV2NonConstEpetraMV(RCP<Xpetra::MultiVector<Scalar,Node> > vec);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static const Epetra_MultiVector&                        MV2EpetraMV(const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& vec);
    static       Epetra_MultiVector&                        MV2NonConstEpetraMV(Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& vec);
#else
    static const Epetra_MultiVector&                        MV2EpetraMV(const Xpetra::MultiVector<Scalar,Node>& vec);
    static       Epetra_MultiVector&                        MV2NonConstEpetraMV(Xpetra::MultiVector<Scalar,Node>& vec);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<const Epetra_CrsMatrix>                      Op2EpetraCrs(RCP<const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > Op);
    static RCP<      Epetra_CrsMatrix>                      Op2NonConstEpetraCrs(RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > Op);
#else
    static RCP<const Epetra_CrsMatrix>                      Op2EpetraCrs(RCP<const Xpetra::Matrix<Scalar,Node> > Op);
    static RCP<      Epetra_CrsMatrix>                      Op2NonConstEpetraCrs(RCP<Xpetra::Matrix<Scalar,Node> > Op);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static const Epetra_CrsMatrix&                          Op2EpetraCrs(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op);
    static       Epetra_CrsMatrix&                          Op2NonConstEpetraCrs(Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op);
#else
    static const Epetra_CrsMatrix&                          Op2EpetraCrs(const Xpetra::Matrix<Scalar,Node>& Op);
    static       Epetra_CrsMatrix&                          Op2NonConstEpetraCrs(Xpetra::Matrix<Scalar,Node>& Op);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static const Epetra_Map&                                Map2EpetraMap(const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>& map);
#else
    static const Epetra_Map&                                Map2EpetraMap(const Xpetra::Map<Node>& map);
#endif
    // @}
#endif

#ifdef HAVE_MUELU_TPETRA
    //! Helper utility to pull out the underlying Tpetra objects from an Xpetra object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<const Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > MV2TpetraMV(RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > const vec);
    static RCP<      Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > MV2NonConstTpetraMV(RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > vec);
    static RCP<      Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > MV2NonConstTpetraMV2(Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& vec);

    static const Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>&      MV2TpetraMV(const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& vec);
    static       Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>&      MV2NonConstTpetraMV(Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& vec);

    static RCP<const Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >   Op2TpetraCrs(RCP<const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > Op);
    static RCP<      Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >   Op2NonConstTpetraCrs(RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > Op);

    static const Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>&        Op2TpetraCrs(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op);
    static       Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>&        Op2NonConstTpetraCrs(Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op);

    static RCP<const Tpetra::RowMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >   Op2TpetraRow(RCP<const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > Op);
    static RCP<      Tpetra::RowMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >   Op2NonConstTpetraRow(RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > Op);


    static const RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node> >        Map2TpetraMap(const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>& map);
#endif

    static RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >          Crs2Op(RCP<Xpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > Op) { return UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Crs2Op(Op); }
    static Teuchos::ArrayRCP<Scalar>                                             GetMatrixDiagonal(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::GetMatrixDiagonal(A); }
    static RCP<Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >          GetMatrixDiagonalInverse(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A, Magnitude tol = Teuchos::ScalarTraits<Scalar>::eps()*100) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::GetMatrixDiagonalInverse(A,tol); }
    static Teuchos::ArrayRCP<Scalar>                                             GetLumpedMatrixDiagonal(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::GetLumpedMatrixDiagonal(A); }
    static Teuchos::RCP<Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > GetLumpedMatrixDiagonal(Teuchos::RCP<const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > A) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::GetLumpedMatrixDiagonal(A); }
    static RCP<Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >          GetMatrixOverlappedDiagonal(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::GetMatrixOverlappedDiagonal(A); }
    static Teuchos::RCP<Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > GetInverse(Teuchos::RCP<const Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > v, Magnitude tol = Teuchos::ScalarTraits<Scalar>::eps()*100, Scalar tolReplacement = Teuchos::ScalarTraits<Scalar>::zero()) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::GetInverse(v,tol,tolReplacement); }
    static Teuchos::Array<Magnitude>                                             ResidualNorm(const Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op, const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& X, const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& RHS) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::ResidualNorm(Op,X,RHS); }
    static Teuchos::Array<Magnitude>                                             ResidualNorm(const Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op, const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& X, const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& RHS, Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Resid) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::ResidualNorm(Op,X,RHS,Resid); }
    static RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >     Residual(const Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op, const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& X, const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& RHS) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Residual(Op,X,RHS); }
    static void Residual(const Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op,  const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& X,  const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& RHS, Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Resid) { MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Residual(Op,X,RHS,Resid);}
    static void                                                                  PauseForDebugger() { MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::PauseForDebugger(); }
    static RCP<Teuchos::FancyOStream>                                            MakeFancy(std::ostream& os) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::MakeFancy(os); }
    static typename Teuchos::ScalarTraits<Scalar>::magnitudeType                 Distance2(const Teuchos::Array<Teuchos::ArrayRCP<const Scalar>>& v, LocalOrdinal i0, LocalOrdinal i1) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Distance2(v,i0,i1); }
    static Teuchos::ArrayRCP<const bool>                                         DetectDirichletRows(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A, const Magnitude& tol = Teuchos::ScalarTraits<Scalar>::magnitude(0.), const bool count_twos_as_dirichlet=false) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::DetectDirichletRows(A,tol,count_twos_as_dirichlet); }
    static Teuchos::ArrayRCP<const bool>                                         DetectDirichletRowsExt(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A, bool & bHasZeroDiagonal, const Magnitude& tol = Teuchos::ScalarTraits<Scalar>::zero()) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::DetectDirichletRowsExt(A,bHasZeroDiagonal,tol); }
#else
    static RCP<const Tpetra::MultiVector<Scalar,Node> > MV2TpetraMV(RCP<Xpetra::MultiVector<Scalar,Node> > const vec);
    static RCP<      Tpetra::MultiVector<Scalar,Node> > MV2NonConstTpetraMV(RCP<Xpetra::MultiVector<Scalar,Node> > vec);
    static RCP<      Tpetra::MultiVector<Scalar,Node> > MV2NonConstTpetraMV2(Xpetra::MultiVector<Scalar,Node>& vec);

    static const Tpetra::MultiVector<Scalar,Node>&      MV2TpetraMV(const Xpetra::MultiVector<Scalar,Node>& vec);
    static       Tpetra::MultiVector<Scalar,Node>&      MV2NonConstTpetraMV(Xpetra::MultiVector<Scalar,Node>& vec);

    static RCP<const Tpetra::CrsMatrix<Scalar,Node> >   Op2TpetraCrs(RCP<const Xpetra::Matrix<Scalar,Node> > Op);
    static RCP<      Tpetra::CrsMatrix<Scalar,Node> >   Op2NonConstTpetraCrs(RCP<Xpetra::Matrix<Scalar,Node> > Op);

    static const Tpetra::CrsMatrix<Scalar,Node>&        Op2TpetraCrs(const Xpetra::Matrix<Scalar,Node>& Op);
    static       Tpetra::CrsMatrix<Scalar,Node>&        Op2NonConstTpetraCrs(Xpetra::Matrix<Scalar,Node>& Op);

    static RCP<const Tpetra::RowMatrix<Scalar,Node> >   Op2TpetraRow(RCP<const Xpetra::Matrix<Scalar,Node> > Op);
    static RCP<      Tpetra::RowMatrix<Scalar,Node> >   Op2NonConstTpetraRow(RCP<Xpetra::Matrix<Scalar,Node> > Op);


    static const RCP<const Tpetra::Map<Node> >        Map2TpetraMap(const Xpetra::Map<Node>& map);
#endif

    static RCP<Xpetra::Matrix<Scalar,Node> >          Crs2Op(RCP<Xpetra::CrsMatrix<Scalar,Node> > Op) { return UtilitiesBase<Scalar,Node>::Crs2Op(Op); }
    static Teuchos::ArrayRCP<Scalar>                                             GetMatrixDiagonal(const Xpetra::Matrix<Scalar,Node>& A) { return MueLu::UtilitiesBase<Scalar,Node>::GetMatrixDiagonal(A); }
    static RCP<Xpetra::Vector<Scalar,Node> >          GetMatrixDiagonalInverse(const Xpetra::Matrix<Scalar,Node>& A, Magnitude tol = Teuchos::ScalarTraits<Scalar>::eps()*100) { return MueLu::UtilitiesBase<Scalar,Node>::GetMatrixDiagonalInverse(A,tol); }
    static Teuchos::ArrayRCP<Scalar>                                             GetLumpedMatrixDiagonal(const Xpetra::Matrix<Scalar,Node>& A) { return MueLu::UtilitiesBase<Scalar,Node>::GetLumpedMatrixDiagonal(A); }
    static Teuchos::RCP<Xpetra::Vector<Scalar,Node> > GetLumpedMatrixDiagonal(Teuchos::RCP<const Xpetra::Matrix<Scalar,Node> > A) { return MueLu::UtilitiesBase<Scalar,Node>::GetLumpedMatrixDiagonal(A); }
    static RCP<Xpetra::Vector<Scalar,Node> >          GetMatrixOverlappedDiagonal(const Xpetra::Matrix<Scalar,Node>& A) { return MueLu::UtilitiesBase<Scalar,Node>::GetMatrixOverlappedDiagonal(A); }
    static Teuchos::RCP<Xpetra::Vector<Scalar,Node> > GetInverse(Teuchos::RCP<const Xpetra::Vector<Scalar,Node> > v, Magnitude tol = Teuchos::ScalarTraits<Scalar>::eps()*100, Scalar tolReplacement = Teuchos::ScalarTraits<Scalar>::zero()) { return MueLu::UtilitiesBase<Scalar,Node>::GetInverse(v,tol,tolReplacement); }
    static Teuchos::Array<Magnitude>                                             ResidualNorm(const Xpetra::Operator<Scalar,Node>& Op, const Xpetra::MultiVector<Scalar,Node>& X, const Xpetra::MultiVector<Scalar,Node>& RHS) { return MueLu::UtilitiesBase<Scalar,Node>::ResidualNorm(Op,X,RHS); }
    static Teuchos::Array<Magnitude>                                             ResidualNorm(const Xpetra::Operator<Scalar,Node>& Op, const Xpetra::MultiVector<Scalar,Node>& X, const Xpetra::MultiVector<Scalar,Node>& RHS, Xpetra::MultiVector<Scalar,Node>& Resid) { return MueLu::UtilitiesBase<Scalar,Node>::ResidualNorm(Op,X,RHS,Resid); }
    static RCP<Xpetra::MultiVector<Scalar,Node> >     Residual(const Xpetra::Operator<Scalar,Node>& Op, const Xpetra::MultiVector<Scalar,Node>& X, const Xpetra::MultiVector<Scalar,Node>& RHS) { return MueLu::UtilitiesBase<Scalar,Node>::Residual(Op,X,RHS); }
    static void Residual(const Xpetra::Operator<Scalar,Node>& Op,  const Xpetra::MultiVector<Scalar,Node>& X,  const Xpetra::MultiVector<Scalar,Node>& RHS, Xpetra::MultiVector<Scalar,Node>& Resid) { MueLu::UtilitiesBase<Scalar,Node>::Residual(Op,X,RHS,Resid);}
    static void                                                                  PauseForDebugger() { MueLu::UtilitiesBase<Scalar,Node>::PauseForDebugger(); }
    static RCP<Teuchos::FancyOStream>                                            MakeFancy(std::ostream& os) { return MueLu::UtilitiesBase<Scalar,Node>::MakeFancy(os); }
    static typename Teuchos::ScalarTraits<Scalar>::magnitudeType                 Distance2(const Teuchos::Array<Teuchos::ArrayRCP<const Scalar>>& v, LocalOrdinal i0, LocalOrdinal i1) { return MueLu::UtilitiesBase<Scalar,Node>::Distance2(v,i0,i1); }
    static Teuchos::ArrayRCP<const bool>                                         DetectDirichletRows(const Xpetra::Matrix<Scalar,Node>& A, const Magnitude& tol = Teuchos::ScalarTraits<Scalar>::magnitude(0.), const bool count_twos_as_dirichlet=false) { return MueLu::UtilitiesBase<Scalar,Node>::DetectDirichletRows(A,tol,count_twos_as_dirichlet); }
    static Teuchos::ArrayRCP<const bool>                                         DetectDirichletRowsExt(const Xpetra::Matrix<Scalar,Node>& A, bool & bHasZeroDiagonal, const Magnitude& tol = Teuchos::ScalarTraits<Scalar>::zero()) { return MueLu::UtilitiesBase<Scalar,Node>::DetectDirichletRowsExt(A,bHasZeroDiagonal,tol); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void                                                                  SetRandomSeed(const Teuchos::Comm<int> &comm) { MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::SetRandomSeed(comm); }
#else
    static void                                                                  SetRandomSeed(const Teuchos::Comm<int> &comm) { MueLu::UtilitiesBase<Scalar,Node>::SetRandomSeed(comm); }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static Scalar PowerMethod(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A, bool scaleByDiag = true,
#else
    static Scalar PowerMethod(const Xpetra::Matrix<Scalar,Node>& A, bool scaleByDiag = true,
#endif
                              LocalOrdinal niters = 10, Magnitude tolerance = 1e-2, bool verbose = false, unsigned int seed = 123) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::PowerMethod(A,scaleByDiag,niters,tolerance,verbose,seed);
#else
      return MueLu::UtilitiesBase<Scalar,Node>::PowerMethod(A,scaleByDiag,niters,tolerance,verbose,seed);
#endif
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static Scalar Frobenius(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A, const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& B) {
      return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Frobenius(A, B);
#else
    static Scalar Frobenius(const Xpetra::Matrix<Scalar,Node>& A, const Xpetra::Matrix<Scalar,Node>& B) {
      return MueLu::UtilitiesBase<Scalar,Node>::Frobenius(A, B);
#endif
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MyOldScaleMatrix(Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op, const Teuchos::ArrayRCP<const Scalar>& scalingVector, bool doInverse = true,
#else
    static void MyOldScaleMatrix(Xpetra::Matrix<Scalar,Node>& Op, const Teuchos::ArrayRCP<const Scalar>& scalingVector, bool doInverse = true,
#endif
                                 bool doFillComplete = true, bool doOptimizeStorage = true);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MyOldScaleMatrix_Epetra(Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op, const Teuchos::ArrayRCP<Scalar>& scalingVector,
#else
    static void MyOldScaleMatrix_Epetra(Xpetra::Matrix<Scalar,Node>& Op, const Teuchos::ArrayRCP<Scalar>& scalingVector,
#endif
                                        bool doFillComplete, bool doOptimizeStorage);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MyOldScaleMatrix_Tpetra(Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op, const Teuchos::ArrayRCP<Scalar>& scalingVector,
#else
    static void MyOldScaleMatrix_Tpetra(Xpetra::Matrix<Scalar,Node>& Op, const Teuchos::ArrayRCP<Scalar>& scalingVector,
#endif
                                        bool doFillComplete, bool doOptimizeStorage);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > Transpose(Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op, bool optimizeTranspose = false,const std::string & label = std::string(),const Teuchos::RCP<Teuchos::ParameterList> &params=Teuchos::null);
#else
    static RCP<Xpetra::Matrix<Scalar,Node> > Transpose(Xpetra::Matrix<Scalar,Node>& Op, bool optimizeTranspose = false,const std::string & label = std::string(),const Teuchos::RCP<Teuchos::ParameterList> &params=Teuchos::null);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > RealValuedToScalarMultiVector(RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,LocalOrdinal,GlobalOrdinal,Node> > X);
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> > RealValuedToScalarMultiVector(RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,Node> > X);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,LocalOrdinal,GlobalOrdinal,Node> > ExtractCoordinatesFromParameterList(ParameterList& paramList);
#else
    static RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,Node> > ExtractCoordinatesFromParameterList(ParameterList& paramList);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void FindDirichletRows(Teuchos::RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > & A, std::vector<LocalOrdinal>& dirichletRows,bool count_twos_as_dirichlet=false) {
      MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::FindDirichletRows(A,dirichletRows,count_twos_as_dirichlet);
#else
    static void FindDirichletRows(Teuchos::RCP<Xpetra::Matrix<Scalar,Node> > & A, std::vector<LocalOrdinal>& dirichletRows,bool count_twos_as_dirichlet=false) {
      MueLu::UtilitiesBase<Scalar,Node>::FindDirichletRows(A,dirichletRows,count_twos_as_dirichlet);
#endif
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void ApplyOAZToMatrixRows(Teuchos::RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& A,const std::vector<LocalOrdinal>& dirichletRows) {
      MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::ApplyOAZToMatrixRows(A,dirichletRows);
#else
    static void ApplyOAZToMatrixRows(Teuchos::RCP<Xpetra::Matrix<Scalar,Node> >& A,const std::vector<LocalOrdinal>& dirichletRows) {
      MueLu::UtilitiesBase<Scalar,Node>::ApplyOAZToMatrixRows(A,dirichletRows);
#endif
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void ApplyOAZToMatrixRows(Teuchos::RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& A,const Teuchos::ArrayRCP<const bool>& dirichletRows) {
      MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::ApplyOAZToMatrixRows(A,dirichletRows);
#else
    static void ApplyOAZToMatrixRows(Teuchos::RCP<Xpetra::Matrix<Scalar,Node> >& A,const Teuchos::ArrayRCP<const bool>& dirichletRows) {
      MueLu::UtilitiesBase<Scalar,Node>::ApplyOAZToMatrixRows(A,dirichletRows);
#endif
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void ZeroDirichletRows(Teuchos::RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& A,const std::vector<LocalOrdinal>& dirichletRows, Scalar replaceWith=Teuchos::ScalarTraits<Scalar>::zero()) {
      MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::ZeroDirichletRows(A,dirichletRows,replaceWith);
#else
    static void ZeroDirichletRows(Teuchos::RCP<Xpetra::Matrix<Scalar,Node> >& A,const std::vector<LocalOrdinal>& dirichletRows, Scalar replaceWith=Teuchos::ScalarTraits<Scalar>::zero()) {
      MueLu::UtilitiesBase<Scalar,Node>::ZeroDirichletRows(A,dirichletRows,replaceWith);
#endif
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void ZeroDirichletRows(Teuchos::RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& A,const Teuchos::ArrayRCP<const bool>& dirichletRows, Scalar replaceWith=Teuchos::ScalarTraits<Scalar>::zero()) {
      MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::ZeroDirichletRows(A,dirichletRows,replaceWith);
#else
    static void ZeroDirichletRows(Teuchos::RCP<Xpetra::Matrix<Scalar,Node> >& A,const Teuchos::ArrayRCP<const bool>& dirichletRows, Scalar replaceWith=Teuchos::ScalarTraits<Scalar>::zero()) {
      MueLu::UtilitiesBase<Scalar,Node>::ZeroDirichletRows(A,dirichletRows,replaceWith);
#endif
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void ZeroDirichletRows(Teuchos::RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& X,const Teuchos::ArrayRCP<const bool>& dirichletRows,Scalar replaceWith=Teuchos::ScalarTraits<Scalar>::zero()) {
      MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::ZeroDirichletRows(X,dirichletRows,replaceWith);
#else
    static void ZeroDirichletRows(Teuchos::RCP<Xpetra::MultiVector<Scalar,Node> >& X,const Teuchos::ArrayRCP<const bool>& dirichletRows,Scalar replaceWith=Teuchos::ScalarTraits<Scalar>::zero()) {
      MueLu::UtilitiesBase<Scalar,Node>::ZeroDirichletRows(X,dirichletRows,replaceWith);
#endif
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void ZeroDirichletCols(Teuchos::RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& A,const Teuchos::ArrayRCP<const bool>& dirichletCols, Scalar replaceWith=Teuchos::ScalarTraits<Scalar>::zero()) {
      MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::ZeroDirichletCols(A,dirichletCols,replaceWith);
#else
    static void ZeroDirichletCols(Teuchos::RCP<Xpetra::Matrix<Scalar,Node> >& A,const Teuchos::ArrayRCP<const bool>& dirichletCols, Scalar replaceWith=Teuchos::ScalarTraits<Scalar>::zero()) {
      MueLu::UtilitiesBase<Scalar,Node>::ZeroDirichletCols(A,dirichletCols,replaceWith);
#endif
    }

  }; // class Utilities

  ///////////////////////////////////////////

#ifdef HAVE_MUELU_EPETRA
  /*!
    @class Utilities
    @brief MueLu utility class (specialization SC=double and LO=GO=int).

    This class provides a number of static helper methods. Some are temporary and will eventually
    go away, while others should be moved to Xpetra.

  Note: this is the implementation for Epetra. Tpetra throws if TPETRA_INST_INT_INT is disabled!
  */
  template <>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  class Utilities<double,int,int,Xpetra::EpetraNode> : public UtilitiesBase<double,int,int,Xpetra::EpetraNode> {
#else
  class Utilities<double,Xpetra::EpetraNode> : public UtilitiesBase<double,Xpetra::EpetraNode> {
#endif
  public:
    typedef double              Scalar;
    typedef int                 LocalOrdinal;
    typedef int                 GlobalOrdinal;
    typedef Xpetra::EpetraNode  Node;
    typedef Teuchos::ScalarTraits<Scalar>::magnitudeType Magnitude;

  private:
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Xpetra::CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node> CrsMatrixWrap;
    typedef Xpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> CrsMatrix;
    typedef Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> Matrix;
    typedef Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> Vector;
    typedef Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> MultiVector;
    typedef Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> Map;
#else
    typedef Xpetra::CrsMatrixWrap<Scalar,Node> CrsMatrixWrap;
    typedef Xpetra::CrsMatrix<Scalar,Node> CrsMatrix;
    typedef Xpetra::Matrix<Scalar,Node> Matrix;
    typedef Xpetra::Vector<Scalar,Node> Vector;
    typedef Xpetra::MultiVector<Scalar,Node> MultiVector;
    typedef Xpetra::Map<Node> Map;
#endif
#ifdef HAVE_MUELU_EPETRA
    typedef Xpetra::EpetraMapT<GlobalOrdinal,Node> EpetraMap;
    typedef Xpetra::EpetraMultiVectorT<GlobalOrdinal,Node> EpetraMultiVector;
    typedef Xpetra::EpetraCrsMatrixT<GlobalOrdinal,Node> EpetraCrsMatrix;
#endif
  public:

#ifdef HAVE_MUELU_EPETRA
    //! Helper utility to pull out the underlying Epetra objects from an Xpetra object
    // @{
    static RCP<const Epetra_MultiVector>                    MV2EpetraMV(RCP<MultiVector> const vec) {
      RCP<const EpetraMultiVector > tmpVec = rcp_dynamic_cast<EpetraMultiVector>(vec);
      if (tmpVec == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::MultiVector to Xpetra::EpetraMultiVector failed");
      return tmpVec->getEpetra_MultiVector();
    }
    static RCP<      Epetra_MultiVector>                    MV2NonConstEpetraMV(RCP<MultiVector> vec) {
      RCP<const EpetraMultiVector> tmpVec = rcp_dynamic_cast<EpetraMultiVector>(vec);
      if (tmpVec == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::MultiVector to Xpetra::EpetraMultiVector failed");
      return tmpVec->getEpetra_MultiVector();
    }

    static const Epetra_MultiVector&                        MV2EpetraMV(const MultiVector& vec) {
      const EpetraMultiVector& tmpVec = dynamic_cast<const EpetraMultiVector&>(vec);
      return *(tmpVec.getEpetra_MultiVector());
    }
    static       Epetra_MultiVector&                        MV2NonConstEpetraMV(MultiVector& vec) {
      const EpetraMultiVector& tmpVec = dynamic_cast<const EpetraMultiVector&>(vec);
      return *(tmpVec.getEpetra_MultiVector());
    }

    static RCP<const Epetra_CrsMatrix>                      Op2EpetraCrs(RCP<const Matrix> Op) {
      RCP<const CrsMatrixWrap> crsOp = rcp_dynamic_cast<const CrsMatrixWrap>(Op);
      if (crsOp == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");
      const RCP<const EpetraCrsMatrix>& tmp_ECrsMtx = rcp_dynamic_cast<const EpetraCrsMatrix>(crsOp->getCrsMatrix());
      if (tmp_ECrsMtx == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::CrsMatrix to Xpetra::EpetraCrsMatrix failed");
      return tmp_ECrsMtx->getEpetra_CrsMatrix();
    }
    static RCP<      Epetra_CrsMatrix>                      Op2NonConstEpetraCrs(RCP<Matrix> Op) {
      RCP<const CrsMatrixWrap> crsOp = rcp_dynamic_cast<const CrsMatrixWrap>(Op);
      if (crsOp == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");
      const RCP<const EpetraCrsMatrix> &tmp_ECrsMtx = rcp_dynamic_cast<const EpetraCrsMatrix>(crsOp->getCrsMatrix());
      if (tmp_ECrsMtx == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::CrsMatrix to Xpetra::EpetraCrsMatrix failed");
      return tmp_ECrsMtx->getEpetra_CrsMatrixNonConst();
    }

    static const Epetra_CrsMatrix&                          Op2EpetraCrs(const Matrix& Op) {
      try {
        const CrsMatrixWrap& crsOp = dynamic_cast<const CrsMatrixWrap&>(Op);
        try {
          const EpetraCrsMatrix& tmp_ECrsMtx = dynamic_cast<const EpetraCrsMatrix&>(*crsOp.getCrsMatrix());
          return *tmp_ECrsMtx.getEpetra_CrsMatrix();
        } catch (std::bad_cast&) {
          throw Exceptions::BadCast("Cast from Xpetra::CrsMatrix to Xpetra::EpetraCrsMatrix failed");
        }
      } catch (std::bad_cast&) {
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");
      }
    }
    static       Epetra_CrsMatrix&                          Op2NonConstEpetraCrs(Matrix& Op) {
      try {
        CrsMatrixWrap& crsOp = dynamic_cast<CrsMatrixWrap&>(Op);
        try {
          EpetraCrsMatrix& tmp_ECrsMtx = dynamic_cast<EpetraCrsMatrix&>(*crsOp.getCrsMatrix());
          return *tmp_ECrsMtx.getEpetra_CrsMatrixNonConst();
        } catch (std::bad_cast&) {
          throw Exceptions::BadCast("Cast from Xpetra::CrsMatrix to Xpetra::EpetraCrsMatrix failed");
        }
      } catch (std::bad_cast&) {
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");
      }
    }

    static const Epetra_Map&                                Map2EpetraMap(const Map& map) {
      RCP<const EpetraMap> xeMap = rcp_dynamic_cast<const EpetraMap>(rcpFromRef(map));
      if (xeMap == Teuchos::null)
        throw Exceptions::BadCast("Utilities::Map2EpetraMap : Cast from Xpetra::Map to Xpetra::EpetraMap failed");
      return xeMap->getEpetra_Map();
    }
    // @}
#endif

#ifdef HAVE_MUELU_TPETRA
    //! Helper utility to pull out the underlying Tpetra objects from an Xpetra object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<const Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > MV2TpetraMV(RCP<MultiVector> const vec)   {
#else
    static RCP<const Tpetra::MultiVector<Scalar,Node> > MV2TpetraMV(RCP<MultiVector> const vec)   {
#endif
#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))
      throw Exceptions::RuntimeError("MV2TpetraMV: Tpetra has not been compiled with support for LO=GO=int.");
#else
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<const Xpetra::TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tmpVec = rcp_dynamic_cast<Xpetra::TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(vec);
#else
      RCP<const Xpetra::TpetraMultiVector<Scalar,Node> > tmpVec = rcp_dynamic_cast<Xpetra::TpetraMultiVector<Scalar,Node> >(vec);
#endif
      if (tmpVec == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::MultiVector to Xpetra::TpetraMultiVector failed");
      return tmpVec->getTpetra_MultiVector();
#endif
    }
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<      Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > MV2NonConstTpetraMV(RCP<MultiVector> vec) {
#else
    static RCP<      Tpetra::MultiVector<Scalar,Node> > MV2NonConstTpetraMV(RCP<MultiVector> vec) {
#endif
#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))
      throw Exceptions::RuntimeError("MV2NonConstTpetraMV: Tpetra has not been compiled with support for LO=GO=int.");
#else
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<const Xpetra::TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tmpVec = rcp_dynamic_cast<Xpetra::TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(vec);
#else
      RCP<const Xpetra::TpetraMultiVector<Scalar,Node> > tmpVec = rcp_dynamic_cast<Xpetra::TpetraMultiVector<Scalar,Node> >(vec);
#endif
      if (tmpVec == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::MultiVector to Xpetra::TpetraMultiVector failed");
      return tmpVec->getTpetra_MultiVector();
#endif

    }
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<      Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > MV2NonConstTpetraMV2(MultiVector& vec)    {
#else
    static RCP<      Tpetra::MultiVector<Scalar,Node> > MV2NonConstTpetraMV2(MultiVector& vec)    {
#endif
#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))
      throw Exceptions::RuntimeError("MV2NonConstTpetraMV2: Tpetra has not been compiled with support for LO=GO=int.");
#else
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const Xpetra::TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& tmpVec = dynamic_cast<const Xpetra::TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>&>(vec);
#else
      const Xpetra::TpetraMultiVector<Scalar,Node>& tmpVec = dynamic_cast<const Xpetra::TpetraMultiVector<Scalar,Node>&>(vec);
#endif
      return tmpVec.getTpetra_MultiVector();
#endif
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static const Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>&      MV2TpetraMV(const MultiVector& vec)   {
#else
    static const Tpetra::MultiVector<Scalar,Node>&      MV2TpetraMV(const MultiVector& vec)   {
#endif
#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))
      throw Exceptions::RuntimeError("MV2TpetraMV: Tpetra has not been compiled with support for LO=GO=int.");
#else
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const Xpetra::TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& tmpVec = dynamic_cast<const Xpetra::TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>&>(vec);
#else
      const Xpetra::TpetraMultiVector<Scalar,Node>& tmpVec = dynamic_cast<const Xpetra::TpetraMultiVector<Scalar,Node>&>(vec);
#endif
      return *(tmpVec.getTpetra_MultiVector());
#endif
    }
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static       Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>&      MV2NonConstTpetraMV(MultiVector& vec) {
#else
    static       Tpetra::MultiVector<Scalar,Node>&      MV2NonConstTpetraMV(MultiVector& vec) {
#endif
#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))
      throw Exceptions::RuntimeError("MV2NonConstTpetraMV: Tpetra has not been compiled with support for LO=GO=int.");
#else
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const Xpetra::TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& tmpVec = dynamic_cast<const Xpetra::TpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>&>(vec);
#else
      const Xpetra::TpetraMultiVector<Scalar,Node>& tmpVec = dynamic_cast<const Xpetra::TpetraMultiVector<Scalar,Node>&>(vec);
#endif
      return *(tmpVec.getTpetra_MultiVector());
#endif
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<const Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >   Op2TpetraCrs(RCP<const Matrix> Op)  {
#else
    static RCP<const Tpetra::CrsMatrix<Scalar,Node> >   Op2TpetraCrs(RCP<const Matrix> Op)  {
#endif
#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))
      throw Exceptions::RuntimeError("Op2TpetraCrs: Tpetra has not been compiled with support for LO=GO=int.");
#else
      // Get the underlying Tpetra Mtx
      RCP<const CrsMatrixWrap> crsOp = rcp_dynamic_cast<const CrsMatrixWrap>(Op);
      if (crsOp == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const RCP<const Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tmp_ECrsMtx = rcp_dynamic_cast<const Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(crsOp->getCrsMatrix());
#else
      const RCP<const Xpetra::TpetraCrsMatrix<Scalar,Node> > &tmp_ECrsMtx = rcp_dynamic_cast<const Xpetra::TpetraCrsMatrix<Scalar,Node> >(crsOp->getCrsMatrix());
#endif
      if (tmp_ECrsMtx == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::CrsMatrix to Xpetra::TpetraCrsMatrix failed");
      return tmp_ECrsMtx->getTpetra_CrsMatrix();
#endif
    }
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<      Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >   Op2NonConstTpetraCrs(RCP<Matrix> Op){
#else
    static RCP<      Tpetra::CrsMatrix<Scalar,Node> >   Op2NonConstTpetraCrs(RCP<Matrix> Op){
#endif
#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))
      throw Exceptions::RuntimeError("Op2NonConstTpetraCrs: Tpetra has not been compiled with support for LO=GO=int.");
#else
      RCP<const CrsMatrixWrap> crsOp = rcp_dynamic_cast<const CrsMatrixWrap>(Op);
      if (crsOp == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const RCP<const Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &tmp_ECrsMtx = rcp_dynamic_cast<const Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(crsOp->getCrsMatrix());
#else
      const RCP<const Xpetra::TpetraCrsMatrix<Scalar,Node> > &tmp_ECrsMtx = rcp_dynamic_cast<const Xpetra::TpetraCrsMatrix<Scalar,Node> >(crsOp->getCrsMatrix());
#endif
      if (tmp_ECrsMtx == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::CrsMatrix to Xpetra::TpetraCrsMatrix failed");
      return tmp_ECrsMtx->getTpetra_CrsMatrixNonConst();
#endif
    };

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static const Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>&        Op2TpetraCrs(const Matrix& Op)   {
#else
    static const Tpetra::CrsMatrix<Scalar,Node>&        Op2TpetraCrs(const Matrix& Op)   {
#endif
#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))
      throw Exceptions::RuntimeError("Op2TpetraCrs: Tpetra has not been compiled with support for LO=GO=int.");
#else
      try {
        const CrsMatrixWrap& crsOp = dynamic_cast<const CrsMatrixWrap&>(Op);
        try {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
          const Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& tmp_ECrsMtx = dynamic_cast<const Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>&>(*crsOp.getCrsMatrix());
#else
          const Xpetra::TpetraCrsMatrix<Scalar,Node>& tmp_ECrsMtx = dynamic_cast<const Xpetra::TpetraCrsMatrix<Scalar,Node>&>(*crsOp.getCrsMatrix());
#endif
          return *tmp_ECrsMtx.getTpetra_CrsMatrix();
        } catch (std::bad_cast&) {
          throw Exceptions::BadCast("Cast from Xpetra::CrsMatrix to Xpetra::TpetraCrsMatrix failed");
        }
      } catch (std::bad_cast&) {
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");
      }
#endif
    }
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static       Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>&        Op2NonConstTpetraCrs(Matrix& Op) {
#else
    static       Tpetra::CrsMatrix<Scalar,Node>&        Op2NonConstTpetraCrs(Matrix& Op) {
#endif
#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))
      throw Exceptions::RuntimeError("Op2NonConstTpetraCrs: Tpetra has not been compiled with support for LO=GO=int.");
#else
      try {
        CrsMatrixWrap& crsOp = dynamic_cast<CrsMatrixWrap&>(Op);
        try {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
          Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& tmp_ECrsMtx = dynamic_cast<Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>&>(*crsOp.getCrsMatrix());
#else
          Xpetra::TpetraCrsMatrix<Scalar,Node>& tmp_ECrsMtx = dynamic_cast<Xpetra::TpetraCrsMatrix<Scalar,Node>&>(*crsOp.getCrsMatrix());
#endif
          return *tmp_ECrsMtx.getTpetra_CrsMatrixNonConst();
        } catch (std::bad_cast&) {
          throw Exceptions::BadCast("Cast from Xpetra::CrsMatrix to Xpetra::TpetraCrsMatrix failed");
        }
      } catch (std::bad_cast&) {
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");
      }
#endif
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<const Tpetra::RowMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >   Op2TpetraRow(RCP<const Matrix> Op)   {
#else
    static RCP<const Tpetra::RowMatrix<Scalar,Node> >   Op2TpetraRow(RCP<const Matrix> Op)   {
#endif
#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))
      throw Exceptions::RuntimeError("Op2TpetraRow: Tpetra has not been compiled with support for LO=GO=int.");
#else
      RCP<const CrsMatrixWrap> crsOp = rcp_dynamic_cast<const CrsMatrixWrap>(Op);
      if (crsOp == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");

      RCP<const CrsMatrix> crsMat = crsOp->getCrsMatrix();
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const RCP<const Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tmp_Crs = rcp_dynamic_cast<const Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(crsMat);
      RCP<const Xpetra::TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tmp_BlockCrs;
#else
      const RCP<const Xpetra::TpetraCrsMatrix<Scalar,Node> > tmp_Crs = rcp_dynamic_cast<const Xpetra::TpetraCrsMatrix<Scalar,Node> >(crsMat);
      RCP<const Xpetra::TpetraBlockCrsMatrix<Scalar,Node> > tmp_BlockCrs;
#endif
      if(!tmp_Crs.is_null()) {
        return tmp_Crs->getTpetra_CrsMatrixNonConst();
      }
      else {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        tmp_BlockCrs= rcp_dynamic_cast<const Xpetra::TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(crsMat);
#else
        tmp_BlockCrs= rcp_dynamic_cast<const Xpetra::TpetraBlockCrsMatrix<Scalar,Node> >(crsMat);
#endif
        if (tmp_BlockCrs.is_null())
          throw Exceptions::BadCast("Cast from Xpetra::CrsMatrix to Xpetra::TpetraCrsMatrix and Xpetra::TpetraBlockCrsMatrix failed");
        return tmp_BlockCrs->getTpetra_BlockCrsMatrixNonConst();
      }
#endif
    }
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<      Tpetra::RowMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >   Op2NonConstTpetraRow(RCP<Matrix> Op) {
#else
    static RCP<      Tpetra::RowMatrix<Scalar,Node> >   Op2NonConstTpetraRow(RCP<Matrix> Op) {
#endif
#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))
      throw Exceptions::RuntimeError("Op2NonConstTpetraRow: Tpetra has not been compiled with support for LO=GO=int.");
#else
      RCP<const CrsMatrixWrap> crsOp = rcp_dynamic_cast<const CrsMatrixWrap>(Op);
      if (crsOp == Teuchos::null)
        throw Exceptions::BadCast("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");

      RCP<const CrsMatrix> crsMat = crsOp->getCrsMatrix();
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const RCP<const Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tmp_Crs = rcp_dynamic_cast<const Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(crsMat);
      RCP<const Xpetra::TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tmp_BlockCrs;
#else
      const RCP<const Xpetra::TpetraCrsMatrix<Scalar,Node> > tmp_Crs = rcp_dynamic_cast<const Xpetra::TpetraCrsMatrix<Scalar,Node> >(crsMat);
      RCP<const Xpetra::TpetraBlockCrsMatrix<Scalar,Node> > tmp_BlockCrs;
#endif
      if(!tmp_Crs.is_null()) {
        return tmp_Crs->getTpetra_CrsMatrixNonConst();
      }
      else {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        tmp_BlockCrs= rcp_dynamic_cast<const Xpetra::TpetraBlockCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(crsMat);
#else
        tmp_BlockCrs= rcp_dynamic_cast<const Xpetra::TpetraBlockCrsMatrix<Scalar,Node> >(crsMat);
#endif
        if (tmp_BlockCrs.is_null())
          throw Exceptions::BadCast("Cast from Xpetra::CrsMatrix to Xpetra::TpetraCrsMatrix and Xpetra::TpetraBlockCrsMatrix failed");
        return tmp_BlockCrs->getTpetra_BlockCrsMatrixNonConst();
      }
#endif
    };


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static const RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >          Map2TpetraMap(const Map& map) {
#else
    static const RCP<const Tpetra::Map<Node> >          Map2TpetraMap(const Map& map) {
#endif
#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))
      throw Exceptions::RuntimeError("Map2TpetraMap: Tpetra has not been compiled with support for LO=GO=int.");
#else
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const RCP<const Xpetra::TpetraMap<LocalOrdinal,GlobalOrdinal,Node>>& tmp_TMap = rcp_dynamic_cast<const Xpetra::TpetraMap<LocalOrdinal,GlobalOrdinal,Node> >(rcpFromRef(map));
#else
      const RCP<const Xpetra::TpetraMap<Node>>& tmp_TMap = rcp_dynamic_cast<const Xpetra::TpetraMap<Node> >(rcpFromRef(map));
#endif
      if (tmp_TMap == Teuchos::null)
        throw Exceptions::BadCast("Utilities::Map2TpetraMap : Cast from Xpetra::Map to Xpetra::TpetraMap failed");
      return tmp_TMap->getTpetra_Map();
#endif
    };
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Matrix>                                                           Crs2Op(RCP<CrsMatrix> Op) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Crs2Op(Op); }
    static Teuchos::ArrayRCP<Scalar>                                             GetMatrixDiagonal(const Matrix& A) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::GetMatrixDiagonal(A); }
    static RCP<Vector>                                                           GetMatrixDiagonalInverse(const Matrix& A, Magnitude tol = Teuchos::ScalarTraits<Scalar>::eps()*100) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::GetMatrixDiagonalInverse(A,tol); }
    static Teuchos::ArrayRCP<Scalar>                                             GetLumpedMatrixDiagonal(const Matrix& A) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::GetLumpedMatrixDiagonal(A); }
    static Teuchos::RCP<Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > GetLumpedMatrixDiagonal(Teuchos::RCP<const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > A) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::GetLumpedMatrixDiagonal(A); }
    static RCP<Vector>                                                           GetMatrixOverlappedDiagonal(const Matrix& A) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::GetMatrixOverlappedDiagonal(A); }
    static RCP<Vector>                                                           GetInverse(Teuchos::RCP<const Vector> v, Magnitude tol = Teuchos::ScalarTraits<Scalar>::eps()*100, Scalar tolReplacement = Teuchos::ScalarTraits<Scalar>::zero()) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::GetInverse(v,tol,tolReplacement); }
    static Teuchos::Array<Magnitude>                                             ResidualNorm(const Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op, const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& X, const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& RHS) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::ResidualNorm(Op,X,RHS); }
    static Teuchos::Array<Magnitude>                                             ResidualNorm(const Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op, const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& X, const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& RHS, Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Resid) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::ResidualNorm(Op,X,RHS,Resid); }
    static RCP<MultiVector>                                                      Residual(const Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op, const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& X, const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& RHS) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Residual(Op,X,RHS); }
    static void Residual(const Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op,  const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& X,  const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& RHS, Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Resid) { MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Residual(Op,X,RHS,Resid);}
    static void                                                                  PauseForDebugger() { MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::PauseForDebugger(); }
    static RCP<Teuchos::FancyOStream>                                            MakeFancy(std::ostream& os) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::MakeFancy(os); }
    static Teuchos::ScalarTraits<Scalar>::magnitudeType                 Distance2(const Teuchos::Array<Teuchos::ArrayRCP<const Scalar>>& v, LocalOrdinal i0, LocalOrdinal i1) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Distance2(v,i0,i1); }
    static Teuchos::ArrayRCP<const bool>                                         DetectDirichletRows(const Matrix& A, const Magnitude& tol = Teuchos::ScalarTraits<Scalar>::zero(), const bool count_twos_as_dirichlet=false) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::DetectDirichletRows(A,tol,count_twos_as_dirichlet); }
    static Teuchos::ArrayRCP<const bool>                                         DetectDirichletRowsExt(const Matrix& A, bool & bHasZeroDiagonal, const Magnitude& tol = Teuchos::ScalarTraits<Scalar>::zero()) { return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::DetectDirichletRowsExt(A,bHasZeroDiagonal,tol); }
    static void                                                                  SetRandomSeed(const Teuchos::Comm<int> &comm) { MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::SetRandomSeed(comm); }
#else
    static RCP<Matrix>                                                           Crs2Op(RCP<CrsMatrix> Op) { return MueLu::UtilitiesBase<Scalar,Node>::Crs2Op(Op); }
    static Teuchos::ArrayRCP<Scalar>                                             GetMatrixDiagonal(const Matrix& A) { return MueLu::UtilitiesBase<Scalar,Node>::GetMatrixDiagonal(A); }
    static RCP<Vector>                                                           GetMatrixDiagonalInverse(const Matrix& A, Magnitude tol = Teuchos::ScalarTraits<Scalar>::eps()*100) { return MueLu::UtilitiesBase<Scalar,Node>::GetMatrixDiagonalInverse(A,tol); }
    static Teuchos::ArrayRCP<Scalar>                                             GetLumpedMatrixDiagonal(const Matrix& A) { return MueLu::UtilitiesBase<Scalar,Node>::GetLumpedMatrixDiagonal(A); }
    static Teuchos::RCP<Xpetra::Vector<Scalar,Node> > GetLumpedMatrixDiagonal(Teuchos::RCP<const Xpetra::Matrix<Scalar,Node> > A) { return MueLu::UtilitiesBase<Scalar,Node>::GetLumpedMatrixDiagonal(A); }
    static RCP<Vector>                                                           GetMatrixOverlappedDiagonal(const Matrix& A) { return MueLu::UtilitiesBase<Scalar,Node>::GetMatrixOverlappedDiagonal(A); }
    static RCP<Vector>                                                           GetInverse(Teuchos::RCP<const Vector> v, Magnitude tol = Teuchos::ScalarTraits<Scalar>::eps()*100, Scalar tolReplacement = Teuchos::ScalarTraits<Scalar>::zero()) { return MueLu::UtilitiesBase<Scalar,Node>::GetInverse(v,tol,tolReplacement); }
    static Teuchos::Array<Magnitude>                                             ResidualNorm(const Xpetra::Operator<Scalar,Node>& Op, const Xpetra::MultiVector<Scalar,Node>& X, const Xpetra::MultiVector<Scalar,Node>& RHS) { return MueLu::UtilitiesBase<Scalar,Node>::ResidualNorm(Op,X,RHS); }
    static Teuchos::Array<Magnitude>                                             ResidualNorm(const Xpetra::Operator<Scalar,Node>& Op, const Xpetra::MultiVector<Scalar,Node>& X, const Xpetra::MultiVector<Scalar,Node>& RHS, Xpetra::MultiVector<Scalar,Node>& Resid) { return MueLu::UtilitiesBase<Scalar,Node>::ResidualNorm(Op,X,RHS,Resid); }
    static RCP<MultiVector>                                                      Residual(const Xpetra::Operator<Scalar,Node>& Op, const Xpetra::MultiVector<Scalar,Node>& X, const Xpetra::MultiVector<Scalar,Node>& RHS) { return MueLu::UtilitiesBase<Scalar,Node>::Residual(Op,X,RHS); }
    static void Residual(const Xpetra::Operator<Scalar,Node>& Op,  const Xpetra::MultiVector<Scalar,Node>& X,  const Xpetra::MultiVector<Scalar,Node>& RHS, Xpetra::MultiVector<Scalar,Node>& Resid) { MueLu::UtilitiesBase<Scalar,Node>::Residual(Op,X,RHS,Resid);}
    static void                                                                  PauseForDebugger() { MueLu::UtilitiesBase<Scalar,Node>::PauseForDebugger(); }
    static RCP<Teuchos::FancyOStream>                                            MakeFancy(std::ostream& os) { return MueLu::UtilitiesBase<Scalar,Node>::MakeFancy(os); }
    static Teuchos::ScalarTraits<Scalar>::magnitudeType                 Distance2(const Teuchos::Array<Teuchos::ArrayRCP<const Scalar>>& v, LocalOrdinal i0, LocalOrdinal i1) { return MueLu::UtilitiesBase<Scalar,Node>::Distance2(v,i0,i1); }
    static Teuchos::ArrayRCP<const bool>                                         DetectDirichletRows(const Matrix& A, const Magnitude& tol = Teuchos::ScalarTraits<Scalar>::zero(), const bool count_twos_as_dirichlet=false) { return MueLu::UtilitiesBase<Scalar,Node>::DetectDirichletRows(A,tol,count_twos_as_dirichlet); }
    static Teuchos::ArrayRCP<const bool>                                         DetectDirichletRowsExt(const Matrix& A, bool & bHasZeroDiagonal, const Magnitude& tol = Teuchos::ScalarTraits<Scalar>::zero()) { return MueLu::UtilitiesBase<Scalar,Node>::DetectDirichletRowsExt(A,bHasZeroDiagonal,tol); }
    static void                                                                  SetRandomSeed(const Teuchos::Comm<int> &comm) { MueLu::UtilitiesBase<Scalar,Node>::SetRandomSeed(comm); }
#endif

    static Scalar PowerMethod(const Matrix& A, bool scaleByDiag = true,
                              LocalOrdinal niters = 10, Magnitude tolerance = 1e-2, bool verbose = false, unsigned int seed = 123) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::PowerMethod(A,scaleByDiag,niters,tolerance,verbose,seed);
#else
      return MueLu::UtilitiesBase<Scalar,Node>::PowerMethod(A,scaleByDiag,niters,tolerance,verbose,seed);
#endif
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static Scalar Frobenius(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A, const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& B) {
      return MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Frobenius(A, B);
#else
    static Scalar Frobenius(const Xpetra::Matrix<Scalar,Node>& A, const Xpetra::Matrix<Scalar,Node>& B) {
      return MueLu::UtilitiesBase<Scalar,Node>::Frobenius(A, B);
#endif
    }

    static void MyOldScaleMatrix(Matrix& Op, const Teuchos::ArrayRCP<const Scalar>& scalingVector, bool doInverse = true,
                                 bool doFillComplete = true, bool doOptimizeStorage = true) {
      Scalar one = Teuchos::ScalarTraits<Scalar>::one();
      Teuchos::ArrayRCP<Scalar> sv(scalingVector.size());
      if (doInverse) {
        for (int i = 0; i < scalingVector.size(); ++i)
          sv[i] = one / scalingVector[i];
      } else {
        for (int i = 0; i < scalingVector.size(); ++i)
          sv[i] = scalingVector[i];
      }

      switch (Op.getRowMap()->lib()) {
        case Xpetra::UseTpetra:
          MyOldScaleMatrix_Tpetra(Op, sv, doFillComplete, doOptimizeStorage);
          break;

        case Xpetra::UseEpetra:
          MyOldScaleMatrix_Epetra(Op, sv, doFillComplete, doOptimizeStorage);
          break;

        default:
          throw Exceptions::RuntimeError("Only Epetra and Tpetra matrices can be scaled.");
      }
    }

    // TODO This is the <double,int,int> specialization
    static void MyOldScaleMatrix_Tpetra(Matrix& Op, const Teuchos::ArrayRCP<Scalar>& scalingVector,
                                        bool doFillComplete, bool doOptimizeStorage) {
#ifdef HAVE_MUELU_TPETRA
#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))
      throw Exceptions::RuntimeError("Matrix scaling is not possible because Tpetra has not been compiled with support for LO=GO=int.");
#else
      try {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>& tpOp = Op2NonConstTpetraCrs(Op);
#else
        Tpetra::CrsMatrix<Scalar, Node>& tpOp = Op2NonConstTpetraCrs(Op);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        const RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > rowMap    = tpOp.getRowMap();
        const RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > domainMap = tpOp.getDomainMap();
        const RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > rangeMap  = tpOp.getRangeMap();
#else
        const RCP<const Tpetra::Map<Node> > rowMap    = tpOp.getRowMap();
        const RCP<const Tpetra::Map<Node> > domainMap = tpOp.getDomainMap();
        const RCP<const Tpetra::Map<Node> > rangeMap  = tpOp.getRangeMap();
#endif

        size_t maxRowSize = tpOp.getNodeMaxNumRowEntries();
        if (maxRowSize == Teuchos::as<size_t>(-1)) // hasn't been determined yet
          maxRowSize = 20;

        std::vector<Scalar> scaledVals(maxRowSize);
        if (tpOp.isFillComplete())
          tpOp.resumeFill();

        if (Op.isLocallyIndexed() == true) {
          Teuchos::ArrayView<const LocalOrdinal> cols;
          Teuchos::ArrayView<const Scalar> vals;

          for (size_t i = 0; i < rowMap->getNodeNumElements(); ++i) {
            tpOp.getLocalRowView(i, cols, vals);
            size_t nnz = tpOp.getNumEntriesInLocalRow(i);
            if (nnz > maxRowSize) {
              maxRowSize = nnz;
              scaledVals.resize(maxRowSize);
            }
            for (size_t j = 0; j < nnz; ++j)
              scaledVals[j] = vals[j]*scalingVector[i];

            if (nnz > 0) {
              Teuchos::ArrayView<const Scalar> valview(&scaledVals[0], nnz);
              tpOp.replaceLocalValues(i, cols, valview);
            }
          } //for (size_t i=0; ...

        } else {
          Teuchos::ArrayView<const GlobalOrdinal> cols;
          Teuchos::ArrayView<const Scalar> vals;

          for (size_t i = 0; i < rowMap->getNodeNumElements(); ++i) {
            GlobalOrdinal gid = rowMap->getGlobalElement(i);
            tpOp.getGlobalRowView(gid, cols, vals);
            size_t nnz = tpOp.getNumEntriesInGlobalRow(gid);
            if (nnz > maxRowSize) {
              maxRowSize = nnz;
              scaledVals.resize(maxRowSize);
            }
            // FIXME FIXME FIXME FIXME FIXME FIXME
            for (size_t j = 0; j < nnz; ++j)
              scaledVals[j] = vals[j]*scalingVector[i]; //FIXME i or gid?

            if (nnz > 0) {
              Teuchos::ArrayView<const Scalar> valview(&scaledVals[0], nnz);
              tpOp.replaceGlobalValues(gid, cols, valview);
            }
          } //for (size_t i=0; ...
        }

        if (doFillComplete) {
          if (domainMap == Teuchos::null || rangeMap == Teuchos::null)
            throw Exceptions::RuntimeError("In Utilities::Scaling: cannot fillComplete because the domain and/or range map hasn't been defined");

          RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
          params->set("Optimize Storage",    doOptimizeStorage);
          params->set("No Nonlocal Changes", true);
          Op.fillComplete(Op.getDomainMap(), Op.getRangeMap(), params);
        }
      } catch(...) {
        throw Exceptions::RuntimeError("Only Tpetra::CrsMatrix types can be scaled (Err.1)");
      }
#endif
#else
      throw Exceptions::RuntimeError("Matrix scaling is not possible because Tpetra has not been enabled.");
#endif
    }

    static void MyOldScaleMatrix_Epetra (Matrix& Op, const Teuchos::ArrayRCP<Scalar>& scalingVector, bool /* doFillComplete */, bool /* doOptimizeStorage */) {
#ifdef HAVE_MUELU_EPETRA
      try {
        //const Epetra_CrsMatrix& epOp = Utilities<double,int,int>::Op2NonConstEpetraCrs(Op);
        const Epetra_CrsMatrix& epOp = Op2NonConstEpetraCrs(Op);

        Epetra_Map const &rowMap = epOp.RowMap();
        int nnz;
        double *vals;
        int *cols;

        for (int i = 0; i < rowMap.NumMyElements(); ++i) {
          epOp.ExtractMyRowView(i, nnz, vals, cols);
          for (int j = 0; j < nnz; ++j)
            vals[j] *= scalingVector[i];
        }

      } catch (...){
        throw Exceptions::RuntimeError("Only Epetra_CrsMatrix types can be scaled");
      }
#else
      throw Exceptions::RuntimeError("Matrix scaling is not possible because Epetra has not been enabled.");
#endif // HAVE_MUELU_EPETRA
    }

    /*! @brief Transpose a Xpetra::Matrix

        Note: Currently, an error is thrown if the matrix isn't a Tpetra::CrsMatrix or Epetra_CrsMatrix.
        In principle, however, we could allow any Epetra_RowMatrix because the Epetra transposer does.
    */
    static RCP<Matrix> Transpose(Matrix& Op, bool /* optimizeTranspose */ = false,const std::string & label = std::string(),const Teuchos::RCP<Teuchos::ParameterList> &params=Teuchos::null) {
      switch (Op.getRowMap()->lib()) {
        case Xpetra::UseTpetra: {
#ifdef HAVE_MUELU_TPETRA
#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))
            throw Exceptions::RuntimeError("Utilities::Transpose: Tpetra is not compiled with LO=GO=int. Add TPETRA_INST_INT_INT:BOOL=ON to your configuration!");
#else
            try {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
              const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>& tpetraOp = Utilities<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Op2TpetraCrs(Op);
#else
              const Tpetra::CrsMatrix<Scalar, Node>& tpetraOp = Utilities<Scalar, Node>::Op2TpetraCrs(Op);
#endif

              // Compute the transpose A of the Tpetra matrix tpetraOp.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
              RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > A;
              Tpetra::RowMatrixTransposer<Scalar, LocalOrdinal, GlobalOrdinal, Node> transposer(rcpFromRef(tpetraOp),label);
#else
              RCP<Tpetra::CrsMatrix<Scalar, Node> > A;
              Tpetra::RowMatrixTransposer<Scalar, Node> transposer(rcpFromRef(tpetraOp),label);
#endif

              {
                using Teuchos::ParameterList;
                using Teuchos::rcp;
                RCP<ParameterList> transposeParams = params.is_null () ?
                  rcp (new ParameterList) :
                  rcp (new ParameterList (*params));
                transposeParams->set ("sort", false);
                A = transposer.createTranspose(transposeParams);
              }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
              RCP<Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > AA   = rcp(new Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(A));
#else
              RCP<Xpetra::TpetraCrsMatrix<Scalar, Node> > AA   = rcp(new Xpetra::TpetraCrsMatrix<Scalar, Node>(A));
#endif
              RCP<CrsMatrix>                                                           AAA  = rcp_implicit_cast<CrsMatrix>(AA);
              RCP<Matrix>                                                              AAAA = rcp( new CrsMatrixWrap(AAA));

              if (Op.IsView("stridedMaps"))
                AAAA->CreateView("stridedMaps", Teuchos::rcpFromRef(Op), true/*doTranspose*/);

              return AAAA;
            }
            catch (std::exception& e) {
              std::cout << "threw exception '" << e.what() << "'" << std::endl;
              throw Exceptions::RuntimeError("Utilities::Transpose failed, perhaps because matrix is not a Crs matrix");
            }
#endif
#else
            throw Exceptions::RuntimeError("Utilities::Transpose: Tpetra is not compiled!");
#endif
          }
        case Xpetra::UseEpetra:
          {
#if defined(HAVE_MUELU_EPETRA) && defined(HAVE_MUELU_EPETRAEXT)
            Teuchos::TimeMonitor tm(*Teuchos::TimeMonitor::getNewTimer("ZZ Entire Transpose"));
            // Epetra case
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            Epetra_CrsMatrix& epetraOp = Utilities<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Op2NonConstEpetraCrs(Op);
#else
            Epetra_CrsMatrix& epetraOp = Utilities<Scalar, Node>::Op2NonConstEpetraCrs(Op);
#endif
            EpetraExt::RowMatrix_Transpose transposer;
            Epetra_CrsMatrix * A = dynamic_cast<Epetra_CrsMatrix*>(&transposer(epetraOp));
            transposer.ReleaseTranspose(); // So we can keep A in Muelu...

            RCP<Epetra_CrsMatrix> rcpA(A);
            RCP<EpetraCrsMatrix> AA   = rcp(new EpetraCrsMatrix(rcpA));
            RCP<CrsMatrix>       AAA  = rcp_implicit_cast<CrsMatrix>(AA);
            RCP<Matrix>          AAAA = rcp( new CrsMatrixWrap(AAA));

            if (Op.IsView("stridedMaps"))
              AAAA->CreateView("stridedMaps", Teuchos::rcpFromRef(Op), true/*doTranspose*/);

            return AAAA;
#else
            throw Exceptions::RuntimeError("Epetra (Err. 2)");
#endif
          }
        default:
          throw Exceptions::RuntimeError("Only Epetra and Tpetra matrices can be transposed.");
      }

      TEUCHOS_UNREACHABLE_RETURN(Teuchos::null);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,LocalOrdinal,GlobalOrdinal,Node> >
    RealValuedToScalarMultiVector(RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,LocalOrdinal,GlobalOrdinal,Node> > X) {
      RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > Xscalar = rcp_dynamic_cast<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(X);
#else
    static RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,Node> >
    RealValuedToScalarMultiVector(RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,Node> > X) {
      RCP<Xpetra::MultiVector<Scalar,Node> > Xscalar = rcp_dynamic_cast<Xpetra::MultiVector<Scalar,Node> >(X);
#endif
      return Xscalar;
    }

    /*! @brief Extract coordinates from parameter list and return them in a Xpetra::MultiVector
    */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,LocalOrdinal,GlobalOrdinal,Node> > ExtractCoordinatesFromParameterList(ParameterList& paramList) {
      RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,LocalOrdinal,GlobalOrdinal,Node> > coordinates = Teuchos::null;
#else
    static RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,Node> > ExtractCoordinatesFromParameterList(ParameterList& paramList) {
      RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,Node> > coordinates = Teuchos::null;
#endif

      // check whether coordinates are contained in parameter list
      if(paramList.isParameter ("Coordinates") == false)
        return coordinates;

  #if defined(HAVE_MUELU_TPETRA)
  #if ( defined(EPETRA_HAVE_OMP) && defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT)) || \
      (!defined(EPETRA_HAVE_OMP) && defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))

      // define Tpetra::MultiVector type with Scalar=float only if
      // * ETI is turned off, since then the compiler will instantiate it automatically OR
      // * Tpetra is instantiated on Scalar=float
  #if !defined(HAVE_TPETRA_EXPLICIT_INSTANTIATION) || defined(HAVE_TPETRA_INST_FLOAT)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      typedef Tpetra::MultiVector<float, LocalOrdinal, GlobalOrdinal, Node> tfMV;
#else
      typedef Tpetra::MultiVector<float, Node> tfMV;
#endif
      RCP<tfMV> floatCoords = Teuchos::null;
  #endif

      // define Tpetra::MultiVector type with Scalar=double only if
      // * ETI is turned off, since then the compiler will instantiate it automatically OR
      // * Tpetra is instantiated on Scalar=double
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      typedef Tpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType, LocalOrdinal, GlobalOrdinal, Node> tdMV;
#else
      typedef Tpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,Node> tdMV;
#endif
      RCP<tdMV> doubleCoords = Teuchos::null;
      if (paramList.isType<RCP<tdMV> >("Coordinates")) {
        // Coordinates are stored as a double vector
        doubleCoords = paramList.get<RCP<tdMV> >("Coordinates");
        paramList.remove("Coordinates");
      }
  #if !defined(HAVE_TPETRA_EXPLICIT_INSTANTIATION) || defined(HAVE_TPETRA_INST_FLOAT)
      else if (paramList.isType<RCP<tfMV> >("Coordinates")) {
        // check if coordinates are stored as a float vector
        floatCoords = paramList.get<RCP<tfMV> >("Coordinates");
        paramList.remove("Coordinates");
        doubleCoords = rcp(new tdMV(floatCoords->getMap(), floatCoords->getNumVectors()));
        deep_copy(*doubleCoords, *floatCoords);
      }
  #endif
      // We have the coordinates in a Tpetra double vector
      if(doubleCoords != Teuchos::null) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        coordinates = Teuchos::rcp(new Xpetra::TpetraMultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType, LocalOrdinal, GlobalOrdinal, Node>(doubleCoords));
#else
        coordinates = Teuchos::rcp(new Xpetra::TpetraMultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,Node>(doubleCoords));
#endif
        TEUCHOS_TEST_FOR_EXCEPT(doubleCoords->getNumVectors() != coordinates->getNumVectors());
      }
  #endif // Tpetra instantiated on GO=int and EpetraNode
  #endif // endif HAVE_TPETRA

  #if defined(HAVE_MUELU_EPETRA)
      RCP<Epetra_MultiVector> doubleEpCoords;
      if (paramList.isType<RCP<Epetra_MultiVector> >("Coordinates")) {
        doubleEpCoords = paramList.get<RCP<Epetra_MultiVector> >("Coordinates");
        paramList.remove("Coordinates");
        RCP<Xpetra::EpetraMultiVectorT<GlobalOrdinal,Node> > epCoordinates = Teuchos::rcp(new Xpetra::EpetraMultiVectorT<GlobalOrdinal,Node>(doubleEpCoords));
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        coordinates = rcp_dynamic_cast<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,LocalOrdinal,GlobalOrdinal,Node> >(epCoordinates);
#else
        coordinates = rcp_dynamic_cast<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType,Node> >(epCoordinates);
#endif
        TEUCHOS_TEST_FOR_EXCEPT(doubleEpCoords->NumVectors() != Teuchos::as<int>(coordinates->getNumVectors()));
      }
  #endif

      // check for Xpetra coordinates vector
      if(paramList.isType<decltype(coordinates)>("Coordinates")) {
        coordinates = paramList.get<decltype(coordinates)>("Coordinates");
      }

      TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(coordinates));
      return coordinates;
    }

  }; // class Utilities (specialization SC=double LO=GO=int)

#endif // HAVE_MUELU_EPETRA



  /*! Removes the following non-serializable data (A,P,R,Nullspace,Coordinates) from level-specific sublists from inList
    and moves it to nonSerialList.  Everything else is copied to serialList.  This function returns the level number of the highest level
    for which non-serializable data was provided.
    */
  long ExtractNonSerializableData(const Teuchos::ParameterList& inList, Teuchos::ParameterList& serialList, Teuchos::ParameterList& nonSerialList);


  /*! Tokenizes a (comma)-separated string, removing all leading and trailing whitespace
  WARNING: This routine is not threadsafe on most architectures
  */
  void TokenizeStringAndStripWhiteSpace(const std::string & stream, std::vector<std::string> & tokenList, const char* token = ",");

  /*! Returns true if a parameter name is a valid Muemex custom level variable, e.g. "MultiVector myArray"
  */
  bool IsParamMuemexVariable(const std::string& name);

  /*! Returns true if a parameter name is a valid user custom level variable, e.g. "MultiVector myArray"
  */
  bool IsParamValidVariable(const std::string& name);

#ifdef HAVE_MUELU_EPETRA
  /*! \fn EpetraCrs_To_XpetraMatrix
      @brief Helper function to convert a Epetra::CrsMatrix to an Xpetra::Matrix
      TODO move this function to an Xpetra utility file
    */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
#else
  template <class Scalar, class Node>
  RCP<Xpetra::Matrix<Scalar, Node> >
#endif
  EpetraCrs_To_XpetraMatrix(const Teuchos::RCP<Epetra_CrsMatrix>& A) {
    typedef Xpetra::EpetraCrsMatrixT<GlobalOrdinal, Node>                      XECrsMatrix;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>       XCrsMatrix;
    typedef Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>   XCrsMatrixWrap;
#else
    typedef Xpetra::CrsMatrix<Scalar, Node>       XCrsMatrix;
    typedef Xpetra::CrsMatrixWrap<Scalar, Node>   XCrsMatrixWrap;
#endif

    RCP<XCrsMatrix> Atmp = rcp(new XECrsMatrix(A));
    return rcp(new XCrsMatrixWrap(Atmp));
  }

  /*! \fn EpetraMultiVector_To_XpetraMultiVector
    @brief Helper function to convert a Epetra::MultiVector to an Xpetra::MultiVector
    TODO move this function to an Xpetra utility file
    */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
#else
  template <class Scalar, class Node>
  RCP<Xpetra::MultiVector<Scalar, Node> >
#endif
  EpetraMultiVector_To_XpetraMultiVector(const Teuchos::RCP<Epetra_MultiVector>& V) {
    return rcp(new Xpetra::EpetraMultiVectorT<GlobalOrdinal, Node>(V));
  }
#endif

#ifdef HAVE_MUELU_TPETRA
  /*! \fn TpetraCrs_To_XpetraMatrix
    @brief Helper function to convert a Tpetra::CrsMatrix to an Xpetra::Matrix
    TODO move this function to an Xpetra utility file
    */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
  TpetraCrs_To_XpetraMatrix(const Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& Atpetra) {
    typedef Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> XTCrsMatrix;
    typedef Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>       XCrsMatrix;
    typedef Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>   XCrsMatrixWrap;
#else
  template <class Scalar, class Node>
  RCP<Xpetra::Matrix<Scalar, Node> >
  TpetraCrs_To_XpetraMatrix(const Teuchos::RCP<Tpetra::CrsMatrix<Scalar, Node> >& Atpetra) {
    typedef Xpetra::TpetraCrsMatrix<Scalar, Node> XTCrsMatrix;
    typedef Xpetra::CrsMatrix<Scalar, Node>       XCrsMatrix;
    typedef Xpetra::CrsMatrixWrap<Scalar, Node>   XCrsMatrixWrap;
#endif

    RCP<XCrsMatrix> Atmp = rcp(new XTCrsMatrix(Atpetra));
    return rcp(new XCrsMatrixWrap(Atmp));
  }

  /*! \fn TpetraCrs_To_XpetraMatrix
    @brief Helper function to convert a Tpetra::FECrsMatrix to an Xpetra::Matrix
    TODO move this function to an Xpetra utility file
    */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
  TpetraFECrs_To_XpetraMatrix(const Teuchos::RCP<Tpetra::FECrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& Atpetra) {
    typedef typename Tpetra::FECrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::crs_matrix_type tpetra_crs_matrix_type;
    typedef Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> XTCrsMatrix;
    typedef Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>       XCrsMatrix;
    typedef Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>   XCrsMatrixWrap;
#else
  template <class Scalar, class Node>
  RCP<Xpetra::Matrix<Scalar, Node> >
  TpetraFECrs_To_XpetraMatrix(const Teuchos::RCP<Tpetra::FECrsMatrix<Scalar, Node> >& Atpetra) {
    typedef typename Tpetra::FECrsMatrix<Scalar, Node>::crs_matrix_type tpetra_crs_matrix_type;
    typedef Xpetra::TpetraCrsMatrix<Scalar, Node> XTCrsMatrix;
    typedef Xpetra::CrsMatrix<Scalar, Node>       XCrsMatrix;
    typedef Xpetra::CrsMatrixWrap<Scalar, Node>   XCrsMatrixWrap;
#endif

    RCP<XCrsMatrix> Atmp = rcp(new XTCrsMatrix(rcp_dynamic_cast<tpetra_crs_matrix_type>(Atpetra)));
    return rcp(new XCrsMatrixWrap(Atmp));
  }

  /*! \fn TpetraMultiVector_To_XpetraMultiVector
    @brief Helper function to convert a Tpetra::MultiVector to an Xpetra::MultiVector
    TODO move this function to an Xpetra utility file
    */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
  TpetraMultiVector_To_XpetraMultiVector(const Teuchos::RCP<Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& Vtpetra) {
    return rcp(new Xpetra::TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>(Vtpetra));
#else
  template <class Scalar, class Node>
  RCP<Xpetra::MultiVector<Scalar, Node> >
  TpetraMultiVector_To_XpetraMultiVector(const Teuchos::RCP<Tpetra::MultiVector<Scalar, Node> >& Vtpetra) {
    return rcp(new Xpetra::TpetraMultiVector<Scalar, Node>(Vtpetra));
#endif
  }

  /*! \fn TpetraFEMultiVector_To_XpetraMultiVector
  @brief Helper function to convert a Tpetra::FEMultiVector to an Xpetra::MultiVector
    TODO move this function to an Xpetra utility file
    */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
  TpetraFEMultiVector_To_XpetraMultiVector(const Teuchos::RCP<Tpetra::FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& Vtpetra) {
    typedef Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> MV;
#else
  template <class Scalar, class Node>
  RCP<Xpetra::MultiVector<Scalar, Node> >
  TpetraFEMultiVector_To_XpetraMultiVector(const Teuchos::RCP<Tpetra::FEMultiVector<Scalar, Node> >& Vtpetra) {
    typedef Tpetra::MultiVector<Scalar, Node> MV;
#endif
    RCP<const MV> Vmv = Teuchos::rcp_dynamic_cast<const MV>(Vtpetra);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    return rcp(new Xpetra::TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>(Vmv));
#else
    return rcp(new Xpetra::TpetraMultiVector<Scalar, Node>(Vmv));
#endif
  }

#endif

  //! Little helper function to convert non-string types to strings
  template<class T>
  std::string toString(const T& what) {
    std::ostringstream buf;
    buf << what;
    return buf.str();
  }

#ifdef HAVE_MUELU_EPETRA
  /*! \fn EpetraCrs_To_XpetraMatrix
    @brief Helper function to convert a Epetra::CrsMatrix to an Xpetra::Matrix
    TODO move this function to an Xpetra utility file
    */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
#else
  template <class Scalar, class Node>
  RCP<Xpetra::Matrix<Scalar, Node> >
#endif
  EpetraCrs_To_XpetraMatrix(const Teuchos::RCP<Epetra_CrsMatrix>& A);

  /*! \fn EpetraMultiVector_To_XpetraMultiVector
    @brief Helper function to convert a Epetra::MultiVector to an Xpetra::MultiVector
    TODO move this function to an Xpetra utility file
    */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
#else
  template <class Scalar, class Node>
  RCP<Xpetra::MultiVector<Scalar, Node> >
#endif
  EpetraMultiVector_To_XpetraMultiVector(const Teuchos::RCP<Epetra_MultiVector>& V);
#endif

#ifdef HAVE_MUELU_TPETRA
  /*! \fn TpetraCrs_To_XpetraMatrix
    @brief Helper function to convert a Tpetra::CrsMatrix to an Xpetra::Matrix
    TODO move this function to an Xpetra utility file
    */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
  TpetraCrs_To_XpetraMatrix(const Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& Atpetra);
#else
  template <class Scalar, class Node>
  RCP<Xpetra::Matrix<Scalar, Node> >
  TpetraCrs_To_XpetraMatrix(const Teuchos::RCP<Tpetra::CrsMatrix<Scalar, Node> >& Atpetra);
#endif

  /*! \fn TpetraMultiVector_To_XpetraMultiVector
    @brief Helper function to convert a Tpetra::MultiVector to an Xpetra::MultiVector
    TODO move this function to an Xpetra utility file
    */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
  TpetraMultiVector_To_XpetraMultiVector(const Teuchos::RCP<Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& Vtpetra);
#else
  template <class Scalar, class Node>
  RCP<Xpetra::MultiVector<Scalar, Node> >
  TpetraMultiVector_To_XpetraMultiVector(const Teuchos::RCP<Tpetra::MultiVector<Scalar, Node> >& Vtpetra);
#endif
#endif

  // Generates a communicator whose only members are other ranks of the baseComm on my node
  Teuchos::RCP<const Teuchos::Comm<int> > GenerateNodeComm(RCP<const Teuchos::Comm<int> > & baseComm, int &NodeId, const int reductionFactor);

} //namespace MueLu

#define MUELU_UTILITIES_SHORT
#endif // MUELU_UTILITIES_DECL_HPP
