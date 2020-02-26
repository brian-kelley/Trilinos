#ifndef MUELU_CREATE_TPETRA_PRECONDITIONER_HPP
#define MUELU_CREATE_TPETRA_PRECONDITIONER_HPP

//! @file
//! @brief Various adapters that will create a MueLu preconditioner that is a Tpetra::Operator.

#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_RowMatrix.hpp>
#include <Xpetra_TpetraBlockCrsMatrix.hpp>
#include <Tpetra_BlockCrsMatrix.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_MultiVector.hpp>
#include <Xpetra_MultiVectorFactory.hpp>

#include <MueLu.hpp>

#include <MueLu_Exceptions.hpp>
#include <MueLu_Hierarchy.hpp>
#include <MueLu_MasterList.hpp>
#include <MueLu_MLParameterListInterpreter.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateXpetraPreconditioner.hpp>
#include <MueLu_Utilities.hpp>
#include <MueLu_HierarchyUtils.hpp>


#if defined(HAVE_MUELU_AMGX)
#include <MueLu_AMGXOperator.hpp>
#include <amgx_c.h>
#include "cuda_runtime.h"
#endif

namespace MueLu {


  /*!
    @brief Helper function to create a MueLu or AMGX preconditioner that can be used by Tpetra.
    @ingroup MueLuAdapters
    Given a Tpetra::Operator, this function returns a constructed MueLu preconditioner.
    @param[in] inA Matrix
    @param[in] inParamList Parameter list
  */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  Teuchos::RCP<MueLu::TpetraOperator<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
  CreateTpetraPreconditioner(const Teuchos::RCP<Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &inA,
#else
  template <class Scalar, class Node>
  Teuchos::RCP<MueLu::TpetraOperator<Scalar,Node> >
  CreateTpetraPreconditioner(const Teuchos::RCP<Tpetra::Operator<Scalar, Node> > &inA,
#endif
                             Teuchos::ParameterList& inParamList)
  {
    typedef Scalar          SC;
    typedef LocalOrdinal    LO;
    typedef GlobalOrdinal   GO;
    typedef Node            NO;

    using   Teuchos::ParameterList;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Xpetra::MultiVector<SC,LO,GO,NO>            MultiVector;
    typedef Xpetra::Matrix<SC,LO,GO,NO>                 Matrix;
    typedef Hierarchy<SC,LO,GO,NO>                      Hierarchy;
    typedef Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> crs_matrix_type;
    typedef Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> block_crs_matrix_type;
#else
    typedef Xpetra::MultiVector<SC,NO>            MultiVector;
    typedef Xpetra::Matrix<SC,NO>                 Matrix;
    typedef Hierarchy<SC,NO>                      Hierarchy;
    typedef Tpetra::CrsMatrix<Scalar, Node> crs_matrix_type;
    typedef Tpetra::BlockCrsMatrix<Scalar, Node> block_crs_matrix_type;
#endif

#if defined(HAVE_MUELU_AMGX)
    std::string externalMG = "use external multigrid package";
    if (inParamList.isParameter(externalMG) && inParamList.get<std::string>(externalMG) == "amgx"){
      const RCP<crs_matrix_type> constCrsA = rcp_dynamic_cast<crs_matrix_type>(inA);
      TEUCHOS_TEST_FOR_EXCEPTION(constCrsA == Teuchos::null, Exceptions::RuntimeError, "CreateTpetraPreconditioner: failed to dynamic cast to Tpetra::CrsMatrix, which is required to be able to use AmgX.");
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      return rcp(new AMGXOperator<SC,LO,GO,NO>(constCrsA,inParamList));
#else
      return rcp(new AMGXOperator<SC,NO>(constCrsA,inParamList));
#endif
    }
#endif

    // Wrap A
    RCP<Matrix> A;
    RCP<block_crs_matrix_type> bcrsA = rcp_dynamic_cast<block_crs_matrix_type>(inA);
    RCP<crs_matrix_type> crsA = rcp_dynamic_cast<crs_matrix_type>(inA);
    if (crsA != Teuchos::null)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      A = TpetraCrs_To_XpetraMatrix<SC,LO,GO,NO>(crsA);
#else
      A = TpetraCrs_To_XpetraMatrix<SC,NO>(crsA);
#endif
    else if (bcrsA != Teuchos::null) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Xpetra::CrsMatrix<SC,LO,GO,NO> > temp = rcp(new Xpetra::TpetraBlockCrsMatrix<SC,LO,GO,NO>(bcrsA));
#else
      RCP<Xpetra::CrsMatrix<SC,NO> > temp = rcp(new Xpetra::TpetraBlockCrsMatrix<SC,NO>(bcrsA));
#endif
      TEUCHOS_TEST_FOR_EXCEPTION(temp==Teuchos::null, Exceptions::RuntimeError, "CreateTpetraPreconditioner: cast from Tpetra::BlockCrsMatrix to Xpetra::TpetraBlockCrsMatrix failed.");
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      A = rcp(new Xpetra::CrsMatrixWrap<SC,LO,GO,NO>(temp));
#else
      A = rcp(new Xpetra::CrsMatrixWrap<SC,NO>(temp));
#endif
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Exceptions::RuntimeError, "CreateTpetraPreconditioner: only Tpetra CrsMatrix and BlockCrsMatrix types are supported.");
    }

    Teuchos::ParameterList& userList = inParamList.sublist("user data");
    if (userList.isParameter("Coordinates")) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::coordinateType,LO,GO,NO> > coordinates = Teuchos::null;
#else
      RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::coordinateType,NO> > coordinates = Teuchos::null;
#endif
      try {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        coordinates = TpetraMultiVector_To_XpetraMultiVector<typename Teuchos::ScalarTraits<Scalar>::coordinateType,LO,GO,NO>(userList.get<RCP<Tpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::coordinateType, LocalOrdinal, GlobalOrdinal, Node> > >("Coordinates"));
#else
        coordinates = TpetraMultiVector_To_XpetraMultiVector<typename Teuchos::ScalarTraits<Scalar>::coordinateType,NO>(userList.get<RCP<Tpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::coordinateType,Node> > >("Coordinates"));
#endif
      } catch(Teuchos::Exceptions::InvalidParameterType&) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        coordinates = userList.get<RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::coordinateType, LocalOrdinal, GlobalOrdinal, Node> > >("Coordinates");
#else
        coordinates = userList.get<RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::coordinateType,Node> > >("Coordinates");
#endif
      }
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      userList.set<RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::coordinateType,LO,GO,NO> > >("Coordinates", coordinates);
#else
      userList.set<RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::coordinateType,NO> > >("Coordinates", coordinates);
#endif
    }

    if (userList.isParameter("Nullspace")) {
      RCP<MultiVector> nullspace = Teuchos::null;
      try {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        nullspace = TpetraMultiVector_To_XpetraMultiVector<SC,LO,GO,NO>(userList.get<RCP<Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> > >("Nullspace"));
#else
        nullspace = TpetraMultiVector_To_XpetraMultiVector<SC,NO>(userList.get<RCP<Tpetra::MultiVector<Scalar, Node> > >("Nullspace"));
#endif
      } catch(Teuchos::Exceptions::InvalidParameterType&) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        nullspace = userList.get<RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> > >("Nullspace");
#else
        nullspace = userList.get<RCP<Xpetra::MultiVector<Scalar, Node> > >("Nullspace");
#endif
      }
      userList.set<RCP<MultiVector> >("Nullspace", nullspace);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Hierarchy> H = MueLu::CreateXpetraPreconditioner<SC,LO,GO,NO>(A, inParamList);
    return rcp(new TpetraOperator<SC,LO,GO,NO>(H));
#else
    RCP<Hierarchy> H = MueLu::CreateXpetraPreconditioner<SC,NO>(A, inParamList);
    return rcp(new TpetraOperator<SC,NO>(H));
#endif
  }


  /*!
    @brief Helper function to create a MueLu preconditioner that can be used by Tpetra.
    @ingroup MueLuAdapters

    Given a Tpetra::Operator, this function returns a constructed MueLu preconditioner.

    @param[in] inA Matrix
    @param[in] xmlFileName XML file containing MueLu options
  */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  Teuchos::RCP<MueLu::TpetraOperator<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
  CreateTpetraPreconditioner(const Teuchos::RCP<Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& inA,
#else
  template <class Scalar, class Node>
  Teuchos::RCP<MueLu::TpetraOperator<Scalar,Node> >
  CreateTpetraPreconditioner(const Teuchos::RCP<Tpetra::Operator<Scalar, Node> >& inA,
#endif
                             const std::string& xmlFileName)
  {
    Teuchos::ParameterList paramList;
    Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, Teuchos::Ptr<Teuchos::ParameterList>(&paramList), *inA->getDomainMap()->getComm());
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    return CreateTpetraPreconditioner<Scalar, LocalOrdinal, GlobalOrdinal, Node>(inA, paramList);
#else
    return CreateTpetraPreconditioner<Scalar, Node>(inA, paramList);
#endif
  }


  /*!
    @brief Helper function to create a MueLu preconditioner that can be used by Tpetra.
    @ingroup MueLuAdapters

    Given a Tpetra::Operator, this function returns a constructed MueLu preconditioner.

    @param[in] inA Matrix
  */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  Teuchos::RCP<MueLu::TpetraOperator<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
  CreateTpetraPreconditioner(const Teuchos::RCP<Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& inA)
#else
  template <class Scalar, class Node>
  Teuchos::RCP<MueLu::TpetraOperator<Scalar,Node> >
  CreateTpetraPreconditioner(const Teuchos::RCP<Tpetra::Operator<Scalar, Node> >& inA)
#endif
  {
    Teuchos::ParameterList paramList;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    return CreateTpetraPreconditioner<Scalar, LocalOrdinal, GlobalOrdinal, Node>(inA, paramList);
#else
    return CreateTpetraPreconditioner<Scalar, Node>(inA, paramList);
#endif
  }


  /*!
    @brief Helper function to reuse an existing MueLu preconditioner.
    @ingroup MueLuAdapters

    @param[in] inA Matrix
    @param[in] Op  Existing MueLu preconditioner.
  */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void ReuseTpetraPreconditioner(const Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& inA,
                                 MueLu::TpetraOperator<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op) {
#else
  template <class Scalar, class Node>
  void ReuseTpetraPreconditioner(const Teuchos::RCP<Tpetra::CrsMatrix<Scalar, Node> >& inA,
                                 MueLu::TpetraOperator<Scalar,Node>& Op) {
#endif
    typedef Scalar          SC;
    typedef LocalOrdinal    LO;
    typedef GlobalOrdinal   GO;
    typedef Node            NO;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Xpetra::Matrix<SC,LO,GO,NO>     Matrix;
    typedef MueLu ::Hierarchy<SC,LO,GO,NO>  Hierarchy;
#else
    typedef Xpetra::Matrix<SC,NO>     Matrix;
    typedef MueLu ::Hierarchy<SC,NO>  Hierarchy;
#endif

    RCP<Hierarchy> H = Op.GetHierarchy();
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Matrix>    A = TpetraCrs_To_XpetraMatrix<SC,LO,GO,NO>(inA);
#else
    RCP<Matrix>    A = TpetraCrs_To_XpetraMatrix<SC,NO>(inA);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MueLu::ReuseXpetraPreconditioner<SC,LO,GO,NO>(A, H);
#else
    MueLu::ReuseXpetraPreconditioner<SC,NO>(A, H);
#endif
  }



} //namespace

#endif //ifndef MUELU_CREATE_TPETRA_PRECONDITIONER_HPP

