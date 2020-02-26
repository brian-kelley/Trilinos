#ifndef MUELU_CREATE_EPETRA_PRECONDITIONER_CPP
#define MUELU_CREATE_EPETRA_PRECONDITIONER_CPP

#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_MultiVector.hpp>
#include <Xpetra_MultiVectorFactory.hpp>

#include <MueLu.hpp>

#include <MueLu_EpetraOperator.hpp>
#include <MueLu_Exceptions.hpp>
#include <MueLu_Hierarchy.hpp>
#include <MueLu_CreateXpetraPreconditioner.hpp>
#include <MueLu_MasterList.hpp>
#include <MueLu_MLParameterListInterpreter.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
#include <MueLu_Utilities.hpp>
#include <MueLu_HierarchyUtils.hpp>

//! @file
//! @brief Various adapters that will create a MueLu preconditioner that is an Epetra_Operator.
#if defined(HAVE_MUELU_EPETRA)
namespace MueLu {

  /*!
    @brief Helper function to create a MueLu preconditioner that can be used by Epetra.
    @ingroup MueLuAdapters
    Given a EpetraCrs_Matrix, this function returns a constructed MueLu preconditioner.
    @param[in] inA Matrix
    @param[in] paramListIn Parameter list
    */
  Teuchos::RCP<MueLu::EpetraOperator>
  CreateEpetraPreconditioner(const Teuchos::RCP<Epetra_CrsMatrix>&   inA,
                             // FIXME: why is it non-const
                             Teuchos::ParameterList& paramListIn)
  {
    using SC = double;
    using LO = int;
    using GO = int;
    using NO = Xpetra::EpetraNode;

    using Teuchos::ParameterList;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    using MultiVector      = Xpetra::MultiVector<SC, LO, GO, NO>;
    using Matrix           = Xpetra::Matrix<SC, LO, GO, NO>;
    using Hierarchy        = Hierarchy<SC,LO,GO,NO>;
    using HierarchyManager = HierarchyManager<SC,LO,GO,NO>;
#else
    using MultiVector      = Xpetra::MultiVector<SC, NO>;
    using Matrix           = Xpetra::Matrix<SC, NO>;
    using Hierarchy        = Hierarchy<SC,NO>;
    using HierarchyManager = HierarchyManager<SC,NO>;
#endif

    Teuchos::ParameterList& userList = paramListIn.sublist("user data");
    if (userList.isParameter("Coordinates")) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<SC>::coordinateType,LO,GO,NO> > coordinates = Teuchos::null;
#else
      RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<SC>::coordinateType,NO> > coordinates = Teuchos::null;
#endif
      try {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        coordinates = EpetraMultiVector_To_XpetraMultiVector<typename Teuchos::ScalarTraits<SC>::coordinateType,LO,GO,NO>(userList.get<RCP<Epetra_MultiVector> >("Coordinates"));
#else
        coordinates = EpetraMultiVector_To_XpetraMultiVector<typename Teuchos::ScalarTraits<SC>::coordinateType,NO>(userList.get<RCP<Epetra_MultiVector> >("Coordinates"));
#endif
      } catch(Teuchos::Exceptions::InvalidParameterType&) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        coordinates = userList.get<RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<SC>::coordinateType, LO, GO, NO> > >("Coordinates");
#else
        coordinates = userList.get<RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<SC>::coordinateType,NO> > >("Coordinates");
#endif
      }
      if(Teuchos::nonnull(coordinates)){
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        userList.set<RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<SC>::coordinateType,LO,GO,NO> > >("Coordinates", coordinates);
#else
        userList.set<RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<SC>::coordinateType,NO> > >("Coordinates", coordinates);
#endif
      }
    }
    if (userList.isParameter("Nullspace")) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<SC>::coordinateType,LO,GO,NO> > nullspace = Teuchos::null;
#else
      RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<SC>::coordinateType,NO> > nullspace = Teuchos::null;
#endif
      try {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        nullspace = EpetraMultiVector_To_XpetraMultiVector<SC,LO,GO,NO>(userList.get<RCP<Epetra_MultiVector> >("Nullspace"));
#else
        nullspace = EpetraMultiVector_To_XpetraMultiVector<SC,NO>(userList.get<RCP<Epetra_MultiVector> >("Nullspace"));
#endif
      } catch(Teuchos::Exceptions::InvalidParameterType&) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        nullspace = userList.get<RCP<Xpetra::MultiVector<SC, LO, GO, NO> > >("Nullspace");
#else
        nullspace = userList.get<RCP<Xpetra::MultiVector<SC, NO> > >("Nullspace");
#endif
      }
      if(Teuchos::nonnull(nullspace)){
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        userList.set<RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<SC>::coordinateType,LO,GO,NO> > >("Nullspace", nullspace);
#else
        userList.set<RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<SC>::coordinateType,NO> > >("Nullspace", nullspace);
#endif
      }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Matrix> A = EpetraCrs_To_XpetraMatrix<SC, LO, GO, NO>(inA);
    RCP<Hierarchy> H = MueLu::CreateXpetraPreconditioner<SC,LO,GO,NO>(A, paramListIn);
#else
    RCP<Matrix> A = EpetraCrs_To_XpetraMatrix<SC, NO>(inA);
    RCP<Hierarchy> H = MueLu::CreateXpetraPreconditioner<SC,NO>(A, paramListIn);
#endif
    return rcp(new EpetraOperator(H));
  }

  /*!
    @brief Helper function to create a MueLu preconditioner that can be used by Epetra.
    @ingroup MueLuAdapters
    Given a Epetra_CrsMatrix, this function returns a constructed MueLu preconditioner.
    @param[in] inA Matrix
    @param[in] xmlFileName XML file containing MueLu options.
    */
  Teuchos::RCP<MueLu::EpetraOperator>
  CreateEpetraPreconditioner(const Teuchos::RCP<Epetra_CrsMatrix>  & A,
                             const std::string& xmlFileName)
  {
    Teuchos::ParameterList paramList;
    Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, Teuchos::Ptr<Teuchos::ParameterList>(&paramList), *Xpetra::toXpetra(A->Comm()));

    return CreateEpetraPreconditioner(A, paramList);
  }

  /*!
    @brief Helper function to create a MueLu preconditioner that can be used by Epetra.
    @ingroup MueLuAdapters
    Given a Epetra_CrsMatrix, this function returns a constructed MueLu preconditioner.
    @param[in] inA Matrix.
    */
  Teuchos::RCP<MueLu::EpetraOperator>
  CreateEpetraPreconditioner(const Teuchos::RCP<Epetra_CrsMatrix>  & A)
  {
    Teuchos::ParameterList paramList;
    return CreateEpetraPreconditioner(A, paramList);
  }

  void ReuseEpetraPreconditioner(const Teuchos::RCP<Epetra_CrsMatrix>& inA, MueLu::EpetraOperator& Op) {
    using SC = double;
    using LO = int;
    using GO = int;
    using NO = Xpetra::EpetraNode;

    using Teuchos::ParameterList;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    using Matrix           = Xpetra::Matrix<SC, LO, GO, NO>;
    using Hierarchy        = Hierarchy<SC,LO,GO,NO>;
#else
    using Matrix           = Xpetra::Matrix<SC, NO>;
    using Hierarchy        = Hierarchy<SC,NO>;
#endif

    RCP<Hierarchy> H = Op.GetHierarchy();
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Matrix>    A = EpetraCrs_To_XpetraMatrix<SC,LO,GO,NO>(inA);
#else
    RCP<Matrix>    A = EpetraCrs_To_XpetraMatrix<SC,NO>(inA);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MueLu::ReuseXpetraPreconditioner<SC,LO,GO,NO>(A, H);
#else
    MueLu::ReuseXpetraPreconditioner<SC,NO>(A, H);
#endif
  }


} //namespace
#endif // HAVE_MUELU_SERIAL and HAVE_MUELU_EPETRA

#endif //ifndef MUELU_CREATE_EPETRA_PRECONDITIONER_CPP
