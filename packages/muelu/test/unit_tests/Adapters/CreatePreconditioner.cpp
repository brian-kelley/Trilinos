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
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "MueLu_TestHelpers.hpp"
#include "MueLu_Version.hpp"

#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_MapFactory.hpp>

#include "MueLu_FactoryManagerBase.hpp"
#include "MueLu_Hierarchy.hpp"
#include "MueLu_PFactory.hpp"
#include "MueLu_SaPFactory.hpp"
#include "MueLu_TransPFactory.hpp"
#include "MueLu_RAPFactory.hpp"
#include "MueLu_AmesosSmoother.hpp"
#include "MueLu_TrilinosSmoother.hpp"
#include "MueLu_SmootherFactory.hpp"
#include "MueLu_CoupledAggregationFactory.hpp"
#include "MueLu_TentativePFactory.hpp"
#include "MueLu_AmesosSmoother.hpp"
#include "MueLu_Utilities.hpp"

#ifdef HAVE_MUELU_TPETRA
#include "MueLu_CreateTpetraPreconditioner.hpp"
#include "MueLu_TpetraOperator.hpp"
#endif
#ifdef HAVE_MUELU_EPETRA
#include "MueLu_CreateEpetraPreconditioner.hpp"
#include "MueLu_EpetraOperator.hpp"
#endif

// #endif

namespace MueLuTests {

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(PetraOperator, CreatePreconditioner, Scalar, LocalOrdinal, GlobalOrdinal, Node)
#else
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(PetraOperator, CreatePreconditioner, Scalar, Node)
#endif
  {
#   include "MueLu_UseShortNames.hpp"
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);

    out << "version: " << MueLu::Version() << std::endl;

    using Teuchos::RCP;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef MueLu::Utilities<SC,LO,GO,NO> Utils;
#else
    typedef MueLu::Utilities<SC,NO> Utils;
#endif
    typedef typename Teuchos::ScalarTraits<SC>::magnitudeType real_type;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Xpetra::MultiVector<real_type,LO,GO,NO> RealValuedMultiVector;
#else
    typedef Xpetra::MultiVector<real_type,NO> RealValuedMultiVector;
#endif

    Xpetra::UnderlyingLib          lib  = TestHelpers::Parameters::getLib();
    RCP<const Teuchos::Comm<int> > comm = TestHelpers::Parameters::getDefaultComm();

    std::string xmlFileName = "test.xml";

    if (lib == Xpetra::UseTpetra) {
#if defined(HAVE_MUELU_TPETRA) && defined(HAVE_MUELU_TPETRA_INST_INT_INT)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      typedef Tpetra::CrsMatrix<SC,LO,GO,NO> tpetra_crsmatrix_type;
      typedef Tpetra::Operator<SC,LO,GO,NO> tpetra_operator_type;
      typedef Tpetra::MultiVector<SC,LO,GO,NO> tpetra_multivector_type;
      typedef Xpetra::MultiVector<real_type,LO,GO,NO> dMultiVector;
      typedef Tpetra::MultiVector<real_type,LO,GO,NO> dtpetra_multivector_type;
#else
      typedef Tpetra::CrsMatrix<SC,NO> tpetra_crsmatrix_type;
      typedef Tpetra::Operator<SC,NO> tpetra_operator_type;
      typedef Tpetra::MultiVector<SC,NO> tpetra_multivector_type;
      typedef Xpetra::MultiVector<real_type,NO> dMultiVector;
      typedef Tpetra::MultiVector<real_type,NO> dtpetra_multivector_type;
#endif

      // Matrix
      GO nx = 1000;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Matrix>     Op  = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(nx * comm->getSize(), lib);
#else
      RCP<Matrix>     Op  = TestHelpers::TestFactory<SC, NO>::Build1DPoisson(nx * comm->getSize(), lib);
#endif
      RCP<const Map > map = Op->getRowMap();

      // Normalized RHS
      RCP<MultiVector> RHS1 = MultiVectorFactory::Build(map, 1);
      RHS1->setSeed(846930886);
      RHS1->randomize();
      Teuchos::Array<typename Teuchos::ScalarTraits<SC>::magnitudeType> norms(1);
      RHS1->norm2(norms);
      RHS1->scale(1/norms[0]);

      // Zero initial guess
      RCP<MultiVector> X1   = MultiVectorFactory::Build(Op->getRowMap(), 1);
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());

#if defined(HAVE_MUELU_ZOLTAN) && defined(HAVE_MPI)
      Teuchos::ParameterList galeriList;
      galeriList.set("nx", nx);
      RCP<RealValuedMultiVector> coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<real_type,LO,GO,Map,RealValuedMultiVector>("1D", Op->getRowMap(), galeriList);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<MultiVector> nullspace   = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(Op->getDomainMap(), 1);
#else
      RCP<MultiVector> nullspace   = Xpetra::MultiVectorFactory<SC,NO>::Build(Op->getDomainMap(), 1);
#endif
      nullspace->putScalar(Teuchos::ScalarTraits<SC>::one());

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<tpetra_crsmatrix_type> tpA = MueLu::Utilities<SC,LO,GO,NO>::Op2NonConstTpetraCrs(Op);
#else
      RCP<tpetra_crsmatrix_type> tpA = MueLu::Utilities<SC,NO>::Op2NonConstTpetraCrs(Op);
#endif

      out << "========== Create Preconditioner from xmlFile ==========" << std::endl;
      out << "xmlFileName: " << xmlFileName << std::endl;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<MueLu::TpetraOperator<SC,LO,GO,NO> > tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(RCP<tpetra_operator_type>(tpA), xmlFileName);
#else
      RCP<MueLu::TpetraOperator<SC,NO> > tH = MueLu::CreateTpetraPreconditioner<SC,NO>(RCP<tpetra_operator_type>(tpA), xmlFileName);
#endif
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

#ifdef HAVE_MUELU_DEPRECATED_TESTS

      out << "Testing deprecated method" << std::endl;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(tpA, xmlFileName);
#else
      tH = MueLu::CreateTpetraPreconditioner<SC,NO>(tpA, xmlFileName);
#endif
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

      xmlFileName = "testWithRebalance.xml";

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<dtpetra_multivector_type> tpcoordinates = MueLu::Utilities<real_type,LO,GO,NO>::MV2NonConstTpetraMV(coordinates);
#else
      RCP<dtpetra_multivector_type> tpcoordinates = MueLu::Utilities<real_type,NO>::MV2NonConstTpetraMV(coordinates);
#endif
      RCP<tpetra_multivector_type>  tpnullspace   = Utils::MV2NonConstTpetraMV(nullspace);

      out << "========== Create Preconditioner from xmlFile and coordinates ==========" << std::endl;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(RCP<tpetra_operator_type>(tpA), xmlFileName, tpcoordinates);
#else
      tH = MueLu::CreateTpetraPreconditioner<SC,NO>(RCP<tpetra_operator_type>(tpA), xmlFileName, tpcoordinates);
#endif
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

      out << "Testing deprecated method" << std::endl;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(tpA, xmlFileName, tpcoordinates);
#else
      tH = MueLu::CreateTpetraPreconditioner<SC,NO>(tpA, xmlFileName, tpcoordinates);
#endif
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

      out << "========== Create Preconditioner from xmlFile, coordinates and nullspace ==========" << std::endl;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(RCP<tpetra_operator_type>(tpA), xmlFileName, tpcoordinates, tpnullspace);
#else
      tH = MueLu::CreateTpetraPreconditioner<SC,NO>(RCP<tpetra_operator_type>(tpA), xmlFileName, tpcoordinates, tpnullspace);
#endif
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

      out << "Testing deprecated method" << std::endl;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(tpA, xmlFileName, tpcoordinates, tpnullspace);
#else
      tH = MueLu::CreateTpetraPreconditioner<SC,NO>(tpA, xmlFileName, tpcoordinates, tpnullspace);
#endif
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

      //Test interface with no ParameterList or XML filename provided.
      out << "========== Create Preconditioner from coordinates and nullspace ==========" << std::endl;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(RCP<tpetra_operator_type>(tpA), tpcoordinates, tpnullspace);
#else
      tH = MueLu::CreateTpetraPreconditioner<SC,NO>(RCP<tpetra_operator_type>(tpA), tpcoordinates, tpnullspace);
#endif
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

      out << "Testing deprecated method" << std::endl;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(tpA, tpcoordinates, tpnullspace);
#else
      tH = MueLu::CreateTpetraPreconditioner<SC,NO>(tpA, tpcoordinates, tpnullspace);
#endif
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;
#endif // HAVE_MUELU_DEPRECATED_TESTS

#endif

#else
      std::cout << "Skip PetraOperator::CreatePreconditioner: Tpetra is not available (with GO=int enabled)" << std::endl;
#endif // #if defined(HAVE_MUELU_TPETRA) && defined(HAVE_MUELU_TPETRA_INST_INT_INT)

    } else if (lib == Xpetra::UseEpetra) {
#ifdef HAVE_MUELU_EPETRA

      // Matrix
      GO nx = 1000;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Matrix>     Op  = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(nx * comm->getSize(), lib);
#else
      RCP<Matrix>     Op  = TestHelpers::TestFactory<SC, NO>::Build1DPoisson(nx * comm->getSize(), lib);
#endif
      RCP<const Map > map = Op->getRowMap();

      // Normalized RHS
      RCP<MultiVector> RHS1 = MultiVectorFactory::Build(map, 1);
      RHS1->setSeed(846930886);
      RHS1->randomize();
      Teuchos::Array<typename Teuchos::ScalarTraits<SC>::magnitudeType> norms(1);
      RHS1->norm2(norms);
      RHS1->scale(1/norms[0]);

      // Zero initial guess
      RCP<MultiVector> X1   = MultiVectorFactory::Build(Op->getRowMap(), 1);
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());

#if defined(HAVE_MUELU_ZOLTAN) && defined(HAVE_MPI)
      Teuchos::ParameterList galeriList;
      galeriList.set("nx", nx);
      RCP<RealValuedMultiVector> coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<real_type,LO,GO,Map,RealValuedMultiVector>("1D", Op->getRowMap(), galeriList);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<MultiVector> nullspace   = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(Op->getDomainMap(), 1);
#else
      RCP<MultiVector> nullspace   = Xpetra::MultiVectorFactory<SC,NO>::Build(Op->getDomainMap(), 1);
#endif
      nullspace->putScalar(Teuchos::ScalarTraits<SC>::one());

      RCP<Epetra_CrsMatrix> epA = Utils::Op2NonConstEpetraCrs(Op);

      RCP<MueLu::EpetraOperator> eH = MueLu::CreateEpetraPreconditioner(epA, xmlFileName);

      eH->Apply(*(Utils::MV2EpetraMV(RHS1)), *(Utils::MV2NonConstEpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

      xmlFileName = "testWithRebalance.xml";

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Epetra_MultiVector> epcoordinates = MueLu::Utilities<real_type,LO,GO,NO>::MV2NonConstEpetraMV(coordinates);
#else
      RCP<Epetra_MultiVector> epcoordinates = MueLu::Utilities<real_type,NO>::MV2NonConstEpetraMV(coordinates);
#endif
      RCP<Epetra_MultiVector> epnullspace   = Utils::MV2NonConstEpetraMV(nullspace);

      Teuchos::ParameterList paramList;
      Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, Teuchos::Ptr<Teuchos::ParameterList>(&paramList), *map->getComm());
      Teuchos::ParameterList& userParamList = paramList.sublist("user data");
      userParamList.set<RCP<Epetra_MultiVector> >("Coordinates", epcoordinates);
      userParamList.set<RCP<Epetra_MultiVector> >("Nullspace", epnullspace);

      eH = MueLu::CreateEpetraPreconditioner(epA, paramList);

      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      eH->Apply(*(Utils::MV2EpetraMV(RHS1)), *(Utils::MV2NonConstEpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

#ifdef HAVE_MUELU_DEPRECATED_TESTS

      eH = MueLu::CreateEpetraPreconditioner(epA, xmlFileName, epcoordinates);

      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      eH->Apply(*(Utils::MV2EpetraMV(RHS1)), *(Utils::MV2NonConstEpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

      eH = MueLu::CreateEpetraPreconditioner(epA, xmlFileName, epcoordinates, epnullspace);

      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      eH->Apply(*(Utils::MV2EpetraMV(RHS1)), *(Utils::MV2NonConstEpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;
#endif // HAVE_MUELU_DEPRECATED_TESTS

#endif

#else
      std::cout << "Skip PetraOperator::CreatePreconditioner: Epetra is not available" << std::endl;
#endif

    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, MueLu::Exceptions::InvalidArgument, "Unknown Xpetra lib");
    }

  } //CreatePreconditioner


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(PetraOperator, CreatePreconditioner_XMLOnList, Scalar, LocalOrdinal, GlobalOrdinal, Node)
#else
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(PetraOperator, CreatePreconditioner_XMLOnList, Scalar, Node)
#endif
  {
#   include "MueLu_UseShortNames.hpp"
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);

    out << "version: " << MueLu::Version() << std::endl;

    using Teuchos::RCP;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef MueLu::Utilities<SC,LO,GO,NO> Utils;
#else
    typedef MueLu::Utilities<SC,NO> Utils;
#endif
    typedef typename Teuchos::ScalarTraits<SC>::magnitudeType real_type;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Xpetra::MultiVector<real_type,LO,GO,NO> dMultiVector;
    typedef Xpetra::MultiVector<real_type,LO,GO,NO> RealValuedMultiVector;
#else
    typedef Xpetra::MultiVector<real_type,NO> dMultiVector;
    typedef Xpetra::MultiVector<real_type,NO> RealValuedMultiVector;
#endif

    Xpetra::UnderlyingLib          lib  = TestHelpers::Parameters::getLib();
    RCP<const Teuchos::Comm<int> > comm = TestHelpers::Parameters::getDefaultComm();

    Teuchos::ParameterList mylist;
    mylist.set("xml parameter file","test.xml");

    if (lib == Xpetra::UseTpetra) {
#if defined(HAVE_MUELU_TPETRA) && defined(HAVE_MUELU_TPETRA_INST_INT_INT)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      typedef Tpetra::Operator<SC,LO,GO,NO> tpetra_operator_type;
#else
      typedef Tpetra::Operator<SC,NO> tpetra_operator_type;
#endif

      // Matrix
      GO nx = 1000;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Matrix>     Op  = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(nx * comm->getSize(), lib);
#else
      RCP<Matrix>     Op  = TestHelpers::TestFactory<SC, NO>::Build1DPoisson(nx * comm->getSize(), lib);
#endif
      RCP<const Map > map = Op->getRowMap();

      // Normalized RHS
      RCP<MultiVector> RHS1 = MultiVectorFactory::Build(map, 1);
      RHS1->setSeed(846930886);
      RHS1->randomize();
      Teuchos::Array<typename Teuchos::ScalarTraits<SC>::magnitudeType> norms(1);
      RHS1->norm2(norms);
      RHS1->scale(1/norms[0]);

      // Zero initial guess
      RCP<MultiVector> X1   = MultiVectorFactory::Build(Op->getRowMap(), 1);
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());

#if defined(HAVE_MUELU_ZOLTAN) && defined(HAVE_MPI)
      Teuchos::ParameterList galeriList;
      galeriList.set("nx", nx);
      RCP<RealValuedMultiVector> coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<real_type,LO,GO,Map,RealValuedMultiVector>("1D", Op->getRowMap(), galeriList);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<MultiVector> nullspace   = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(Op->getDomainMap(), 1);
#else
      RCP<MultiVector> nullspace   = Xpetra::MultiVectorFactory<SC,NO>::Build(Op->getDomainMap(), 1);
#endif
      nullspace->putScalar(Teuchos::ScalarTraits<SC>::one());

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Tpetra::CrsMatrix<SC,LO,GO,NO> > tpA = MueLu::Utilities<SC,LO,GO,NO>::Op2NonConstTpetraCrs(Op);
#else
      RCP<Tpetra::CrsMatrix<SC,NO> > tpA = MueLu::Utilities<SC,NO>::Op2NonConstTpetraCrs(Op);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<MueLu::TpetraOperator<SC,LO,GO,NO> > tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(RCP<tpetra_operator_type>(tpA),mylist);
#else
      RCP<MueLu::TpetraOperator<SC,NO> > tH = MueLu::CreateTpetraPreconditioner<SC,NO>(RCP<tpetra_operator_type>(tpA),mylist);
#endif
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

#ifdef HAVE_MUELU_DEPRECATED_TESTS

      out << "Testing deprecated method" << std::endl;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(tpA,mylist);
#else
      tH = MueLu::CreateTpetraPreconditioner<SC,NO>(tpA,mylist);
#endif
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

#endif // HAVE_MUELU_DEPRECATED_TESTS

      mylist.set("xml parameter file","testWithRebalance.xml");

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Tpetra::MultiVector<real_type,LO,GO,NO> > tpcoordinates = MueLu::Utilities<real_type,LO,GO,NO>::MV2NonConstTpetraMV(coordinates);
      RCP<Tpetra::MultiVector<SC,LO,GO,NO> > tpnullspace   = Utils::MV2NonConstTpetraMV(nullspace);
#else
      RCP<Tpetra::MultiVector<real_type,NO> > tpcoordinates = MueLu::Utilities<real_type,NO>::MV2NonConstTpetraMV(coordinates);
      RCP<Tpetra::MultiVector<SC,NO> > tpnullspace   = Utils::MV2NonConstTpetraMV(nullspace);
#endif

      std::string mueluXML = mylist.get("xml parameter file", "");
      Teuchos::ParameterList mueluList;
      Teuchos::updateParametersFromXmlFileAndBroadcast(mueluXML, Teuchos::Ptr<Teuchos::ParameterList>(&mueluList), *map->getComm());
      Teuchos::ParameterList& userParamList = mueluList.sublist("user data");
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      userParamList.set<RCP<Tpetra::MultiVector<real_type,LO,GO,NO> > >("Coordinates", tpcoordinates);
      userParamList.set<RCP<Tpetra::MultiVector<SC,LO,GO,NO> > >("Nullspace", tpnullspace);
      tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(RCP<tpetra_operator_type>(tpA), mueluList);
#else
      userParamList.set<RCP<Tpetra::MultiVector<real_type,NO> > >("Coordinates", tpcoordinates);
      userParamList.set<RCP<Tpetra::MultiVector<SC,NO> > >("Nullspace", tpnullspace);
      tH = MueLu::CreateTpetraPreconditioner<SC,NO>(RCP<tpetra_operator_type>(tpA), mueluList);
#endif
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

#ifdef HAVE_MUELU_DEPRECATED_TESTS

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(RCP<tpetra_operator_type>(tpA), mylist, tpcoordinates);
#else
      tH = MueLu::CreateTpetraPreconditioner<SC,NO>(RCP<tpetra_operator_type>(tpA), mylist, tpcoordinates);
#endif
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

      out << "Testing deprecated method" << std::endl;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(tpA, mylist, tpcoordinates);
#else
      tH = MueLu::CreateTpetraPreconditioner<SC,NO>(tpA, mylist, tpcoordinates);
#endif
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(RCP<tpetra_operator_type>(tpA), mylist, tpcoordinates, tpnullspace);
#else
      tH = MueLu::CreateTpetraPreconditioner<SC,NO>(RCP<tpetra_operator_type>(tpA), mylist, tpcoordinates, tpnullspace);
#endif
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

      out << "Testing deprecated method" << std::endl;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(tpA, mylist, tpcoordinates, tpnullspace);
#else
      tH = MueLu::CreateTpetraPreconditioner<SC,NO>(tpA, mylist, tpcoordinates, tpnullspace);
#endif
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

#endif // HAVE_MUELU_DEPRECATED_TESTS

#endif

#else
      std::cout << "Skip PetraOperator::CreatePreconditioner_XMLOnList: Tpetra is not available (with GO=int enabled)" << std::endl;
#endif // #if defined(HAVE_MUELU_TPETRA) && defined(HAVE_MUELU_TPETRA_INST_INT_INT)

    } else if (lib == Xpetra::UseEpetra) {
#ifdef HAVE_MUELU_EPETRA
      // Matrix
      GO nx = 1000;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Matrix>     Op  = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(nx * comm->getSize(), lib);
#else
      RCP<Matrix>     Op  = TestHelpers::TestFactory<SC, NO>::Build1DPoisson(nx * comm->getSize(), lib);
#endif
      RCP<const Map > map = Op->getRowMap();

      // Normalized RHS
      RCP<MultiVector> RHS1 = MultiVectorFactory::Build(map, 1);
      RHS1->setSeed(846930886);
      RHS1->randomize();
      Teuchos::Array<typename Teuchos::ScalarTraits<SC>::magnitudeType> norms(1);
      RHS1->norm2(norms);
      RHS1->scale(1/norms[0]);

      // Zero initial guess
      RCP<MultiVector> X1   = MultiVectorFactory::Build(Op->getRowMap(), 1);
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());

#if defined(HAVE_MUELU_ZOLTAN) && defined(HAVE_MPI)
      Teuchos::ParameterList galeriList;
      galeriList.set("nx", nx);
      RCP<dMultiVector> coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<real_type,LO,GO,Map,dMultiVector>("1D", Op->getRowMap(), galeriList);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<MultiVector> nullspace   = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(Op->getDomainMap(), 1);
#else
      RCP<MultiVector> nullspace   = Xpetra::MultiVectorFactory<SC,NO>::Build(Op->getDomainMap(), 1);
#endif
      nullspace->putScalar(Teuchos::ScalarTraits<SC>::one());

      RCP<Epetra_CrsMatrix> epA = Utils::Op2NonConstEpetraCrs(Op);

      RCP<MueLu::EpetraOperator> eH = MueLu::CreateEpetraPreconditioner(epA, mylist);

      eH->Apply(*(Utils::MV2EpetraMV(RHS1)), *(Utils::MV2NonConstEpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

      mylist.set("xml parameter file","testWithRebalance.xml");

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Epetra_MultiVector> epcoordinates = MueLu::Utilities<real_type,LO,GO,NO>::MV2NonConstEpetraMV(coordinates);
#else
      RCP<Epetra_MultiVector> epcoordinates = MueLu::Utilities<real_type,NO>::MV2NonConstEpetraMV(coordinates);
#endif
      RCP<Epetra_MultiVector> epnullspace   = Utils::MV2NonConstEpetraMV(nullspace);

      Teuchos::ParameterList paramList = mylist;
      Teuchos::ParameterList& userParamList = paramList.sublist("user data");
      userParamList.set<RCP<Epetra_MultiVector> >("Coordinates", epcoordinates);
      userParamList.set<RCP<Epetra_MultiVector> >("Nullspace", epnullspace);
      eH = MueLu::CreateEpetraPreconditioner(epA, paramList);

#ifdef HAVE_MUELU_DEPRECATED_TESTS

      eH = MueLu::CreateEpetraPreconditioner(epA, mylist, epcoordinates);

      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      eH->Apply(*(Utils::MV2EpetraMV(RHS1)), *(Utils::MV2NonConstEpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

      eH = MueLu::CreateEpetraPreconditioner(epA, mylist, epcoordinates, epnullspace);

      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      eH->Apply(*(Utils::MV2EpetraMV(RHS1)), *(Utils::MV2NonConstEpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;
#endif // HAVE_MUELU_DEPRECATED_TESTS

#endif

#else
      std::cout << "Skip PetraOperator::CreatePreconditioner_XMLOnList: Epetra is not available" << std::endl;
#endif

    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, MueLu::Exceptions::InvalidArgument, "Unknown Xpetra lib");
    }

  } //CreatePreconditioner_XMLOnList


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(PetraOperator, CreatePreconditioner_PDESystem, Scalar, LocalOrdinal, GlobalOrdinal, Node)
#else
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(PetraOperator, CreatePreconditioner_PDESystem, Scalar, Node)
#endif
  {
#   include "MueLu_UseShortNames.hpp"
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);

    out << "version: " << MueLu::Version() << std::endl;

    using Teuchos::RCP;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef MueLu::Utilities<SC,LO,GO,NO> Utils;
#else
    typedef MueLu::Utilities<SC,NO> Utils;
#endif
    typedef typename Teuchos::ScalarTraits<Scalar>::coordinateType real_type;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Xpetra::MultiVector<real_type,LO,GO,NO> dMultiVector;
#else
    typedef Xpetra::MultiVector<real_type,NO> dMultiVector;
#endif

#if defined(HAVE_MUELU_ZOLTAN) && defined(HAVE_MPI)

    Xpetra::UnderlyingLib          lib  = TestHelpers::Parameters::getLib();
    RCP<const Teuchos::Comm<int> > comm = TestHelpers::Parameters::getDefaultComm();

    for (int k = 0; k < 2; k++) {
      std::string xmlFileName;
      if (k == 0) xmlFileName = "testPDE.xml";
      if (k == 1) xmlFileName = "testPDE1.xml";

      if (lib == Xpetra::UseTpetra) {
#if defined(HAVE_MUELU_TPETRA) && defined(HAVE_MUELU_TPETRA_INST_INT_INT)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        typedef Tpetra::Operator<SC,LO,GO,NO> tpetra_operator_type;
#else
        typedef Tpetra::Operator<SC,NO> tpetra_operator_type;
#endif

        int numPDEs=3;

        // Matrix
        GO nx = 972;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        RCP<Matrix>     Op  = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(nx * comm->getSize(), lib);
#else
        RCP<Matrix>     Op  = TestHelpers::TestFactory<SC, NO>::Build1DPoisson(nx * comm->getSize(), lib);
#endif
        RCP<const Map > map = Op->getRowMap();

        Teuchos::ParameterList clist;
        clist.set("nx", (nx * comm->getSize())/numPDEs);
        RCP<const Map>   cmap        = MapFactory::Build(lib, Teuchos::as<size_t>((nx * comm->getSize())/numPDEs), Teuchos::as<int>(0), comm);
        RCP<dMultiVector> coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<real_type,LO,GO,Map,dMultiVector>("1D", cmap, clist);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        RCP<MultiVector> nullspace   = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(Op->getDomainMap(), numPDEs);
#else
        RCP<MultiVector> nullspace   = Xpetra::MultiVectorFactory<SC,NO>::Build(Op->getDomainMap(), numPDEs);
#endif
        if (numPDEs == 1) {
          nullspace->putScalar(Teuchos::ScalarTraits<Scalar>::one());
        } else {
          for (int i = 0; i < numPDEs; i++) {
            Teuchos::ArrayRCP<Scalar> nsData = nullspace->getDataNonConst(i);
            for (int j = 0; j < nsData.size(); j++) {
              GlobalOrdinal GID = Op->getDomainMap()->getGlobalElement(j) - Op->getDomainMap()->getIndexBase();
              if ((GID-i) % numPDEs == 0)
                nsData[j] = Teuchos::ScalarTraits<Scalar>::one();
            }
          }
        }

        // Normalized RHS
        RCP<MultiVector> RHS1 = MultiVectorFactory::Build(Op->getRowMap(), 1);
        RHS1->setSeed(846930886);
        RHS1->randomize();
        Teuchos::Array<typename Teuchos::ScalarTraits<SC>::magnitudeType> norms(1);
        RHS1->norm2(norms);
        RHS1->scale(1/norms[0]);

        // Zero initial guess
        RCP<MultiVector> X1   = MultiVectorFactory::Build(Op->getRowMap(), 1);
        X1->putScalar(Teuchos::ScalarTraits<SC>::zero());

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        RCP<Tpetra::CrsMatrix<SC,LO,GO,NO> >   tpA           = MueLu::Utilities<SC,LO,GO,NO>::Op2NonConstTpetraCrs(Op);
        RCP<Tpetra::MultiVector<real_type,LO,GO,NO> > tpcoordinates = MueLu::Utilities<real_type,LO,GO,NO>::MV2NonConstTpetraMV(coordinates);
        RCP<Tpetra::MultiVector<SC,LO,GO,NO> > tpnullspace   = Utils::MV2NonConstTpetraMV(nullspace);
#else
        RCP<Tpetra::CrsMatrix<SC,NO> >   tpA           = MueLu::Utilities<SC,NO>::Op2NonConstTpetraCrs(Op);
        RCP<Tpetra::MultiVector<real_type,NO> > tpcoordinates = MueLu::Utilities<real_type,NO>::MV2NonConstTpetraMV(coordinates);
        RCP<Tpetra::MultiVector<SC,NO> > tpnullspace   = Utils::MV2NonConstTpetraMV(nullspace);
#endif

        Teuchos::ParameterList paramList;
        Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, Teuchos::Ptr<Teuchos::ParameterList>(&paramList), *tpA->getDomainMap()->getComm());
        Teuchos::ParameterList& userParamList = paramList.sublist("user data");
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        userParamList.set<RCP<Tpetra::MultiVector<real_type,LO,GO,NO> > >("Coordinates", tpcoordinates);
        userParamList.set<RCP<Tpetra::MultiVector<SC,LO,GO,NO> > >("Nullspace", tpnullspace);
        RCP<MueLu::TpetraOperator<SC,LO,GO,NO> > tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(RCP<tpetra_operator_type>(tpA), paramList);
#else
        userParamList.set<RCP<Tpetra::MultiVector<real_type,NO> > >("Coordinates", tpcoordinates);
        userParamList.set<RCP<Tpetra::MultiVector<SC,NO> > >("Nullspace", tpnullspace);
        RCP<MueLu::TpetraOperator<SC,NO> > tH = MueLu::CreateTpetraPreconditioner<SC,NO>(RCP<tpetra_operator_type>(tpA), paramList);
#endif
        tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
        out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
            Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

#ifdef HAVE_MUELU_DEPRECATED_TESTS

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(RCP<tpetra_operator_type>(tpA), xmlFileName, tpcoordinates);
#else
        tH = MueLu::CreateTpetraPreconditioner<SC,NO>(RCP<tpetra_operator_type>(tpA), xmlFileName, tpcoordinates);
#endif
        tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
        out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
            Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

        out << "Testing deprecated method" << std::endl;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(tpA, xmlFileName, tpcoordinates);
#else
        tH = MueLu::CreateTpetraPreconditioner<SC,NO>(tpA, xmlFileName, tpcoordinates);
#endif
        tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
        out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
            Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(RCP<tpetra_operator_type>(tpA), xmlFileName, tpcoordinates, tpnullspace);
#else
        tH = MueLu::CreateTpetraPreconditioner<SC,NO>(RCP<tpetra_operator_type>(tpA), xmlFileName, tpcoordinates, tpnullspace);
#endif
        X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
        tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
        out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
            Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

        out << "Testing deprecated method" << std::endl;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(tpA, xmlFileName, tpcoordinates, tpnullspace);
#else
        tH = MueLu::CreateTpetraPreconditioner<SC,NO>(tpA, xmlFileName, tpcoordinates, tpnullspace);
#endif
        X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
        tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
        out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
            Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

#endif // HAVE_MUELU_DEPRECATED_TESTS
#else
        std::cout << "Skip PetraOperator::CreatePreconditioner_PDESystem: Tpetra is not available (with GO=int enabled)" << std::endl;
#endif // #if defined(HAVE_MUELU_TPETRA) && defined(HAVE_MUELU_TPETRA_INST_INT_INT)

      } else if (lib == Xpetra::UseEpetra) {
#ifdef HAVE_MUELU_EPETRA
        int numPDEs=3;

        // Matrix
        GO nx = 972;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        RCP<Matrix>     Op  = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(nx * comm->getSize(), lib);
#else
        RCP<Matrix>     Op  = TestHelpers::TestFactory<SC, NO>::Build1DPoisson(nx * comm->getSize(), lib);
#endif
        RCP<const Map > map = Op->getRowMap();

        Teuchos::ParameterList clist;
        clist.set("nx", (nx * comm->getSize())/numPDEs);
        RCP<const Map>   cmap        = MapFactory::Build(lib, Teuchos::as<size_t>((nx * comm->getSize())/numPDEs), Teuchos::as<int>(0), comm);
        RCP<dMultiVector> coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<real_type,LO,GO,Map,dMultiVector>("1D", cmap, clist);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        RCP<MultiVector> nullspace   = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(Op->getDomainMap(), numPDEs);
#else
        RCP<MultiVector> nullspace   = Xpetra::MultiVectorFactory<SC,NO>::Build(Op->getDomainMap(), numPDEs);
#endif
        if (numPDEs == 1) {
          nullspace->putScalar(Teuchos::ScalarTraits<Scalar>::one());
        } else {
          for (int i = 0; i < numPDEs; i++) {
            Teuchos::ArrayRCP<Scalar> nsData = nullspace->getDataNonConst(i);
            for (int j = 0; j < nsData.size(); j++) {
              GlobalOrdinal GID = Op->getDomainMap()->getGlobalElement(j) - Op->getDomainMap()->getIndexBase();
              if ((GID-i) % numPDEs == 0)
                nsData[j] = Teuchos::ScalarTraits<Scalar>::one();
            }
          }
        }

        // Normalized RHS
        RCP<MultiVector> RHS1 = MultiVectorFactory::Build(Op->getRowMap(), 1);
        RHS1->setSeed(846930886);
        RHS1->randomize();
        Teuchos::Array<typename Teuchos::ScalarTraits<SC>::magnitudeType> norms(1);
        RHS1->norm2(norms);
        RHS1->scale(1/norms[0]);

        // Zero initial guess
        RCP<MultiVector> X1   = MultiVectorFactory::Build(Op->getRowMap(), 1);
        X1->putScalar(Teuchos::ScalarTraits<SC>::zero());

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        RCP<Epetra_CrsMatrix>   epA           = MueLu::Utilities<SC,LO,GO,NO>::Op2NonConstEpetraCrs(Op);
        RCP<Epetra_MultiVector> epcoordinates = MueLu::Utilities<real_type,LO,GO,NO>::MV2NonConstEpetraMV(coordinates);
#else
        RCP<Epetra_CrsMatrix>   epA           = MueLu::Utilities<SC,NO>::Op2NonConstEpetraCrs(Op);
        RCP<Epetra_MultiVector> epcoordinates = MueLu::Utilities<real_type,NO>::MV2NonConstEpetraMV(coordinates);
#endif
        RCP<Epetra_MultiVector> epnullspace   = Utils::MV2NonConstEpetraMV(nullspace);

        Teuchos::ParameterList paramList;
        Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName,
                                                         Teuchos::Ptr<Teuchos::ParameterList>(&paramList),
                                                         *map->getComm());
        paramList.set("use kokkos refactor", false); // Done to avoid having kokkos factories called with Epetra
        Teuchos::ParameterList& userParamList = paramList.sublist("user data");
        userParamList.set<RCP<Epetra_MultiVector> >("Coordinates", epcoordinates);
        userParamList.set<RCP<Epetra_MultiVector> >("Nullspace",   epnullspace);
        RCP<MueLu::EpetraOperator> eH = MueLu::CreateEpetraPreconditioner(epA, paramList);

        eH->Apply(*(Utils::MV2EpetraMV(RHS1)), *(Utils::MV2NonConstEpetraMV(X1)));
        out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
            Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

#ifdef HAVE_MUELU_DEPRECATED_TESTS

        eH = MueLu::CreateEpetraPreconditioner(epA, xmlFileName, epcoordinates);

        eH->Apply(*(Utils::MV2EpetraMV(RHS1)), *(Utils::MV2NonConstEpetraMV(X1)));
        out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
            Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

        eH = MueLu::CreateEpetraPreconditioner(epA, xmlFileName, epcoordinates, epnullspace);

        X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
        eH->Apply(*(Utils::MV2EpetraMV(RHS1)), *(Utils::MV2NonConstEpetraMV(X1)));
        out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
            Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;
#endif // HAVE_MUELU_DEPRECATED_TESTS

#else
        std::cout << "Skip PetraOperator::CreatePreconditioner_PDESystem: Epetra is not available" << std::endl;
#endif
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, MueLu::Exceptions::InvalidArgument, "Unknown Xpetra lib");
      }
    }
#endif // defined(HAVE_MUELU_ZOLTAN) && defined(HAVE_MPI)
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(PetraOperator, ReusePreconditioner, Scalar, LocalOrdinal, GlobalOrdinal, Node)
#else
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(PetraOperator, ReusePreconditioner, Scalar, Node)
#endif
  {
#   include "MueLu_UseShortNames.hpp"
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);

    out << "version: " << MueLu::Version() << std::endl;

    using Teuchos::RCP;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef MueLu::Utilities<SC,LO,GO,NO> Utils;
#else
    typedef MueLu::Utilities<SC,NO> Utils;
#endif
    typedef typename Teuchos::ScalarTraits<Scalar>::magnitudeType real_type;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Xpetra::MultiVector<real_type,LO,GO,NO> dMultiVector;
#else
    typedef Xpetra::MultiVector<real_type,NO> dMultiVector;
#endif

    Xpetra::UnderlyingLib          lib  = TestHelpers::Parameters::getLib();
    RCP<const Teuchos::Comm<int> > comm = TestHelpers::Parameters::getDefaultComm();

    std::string xmlFileName = "testReuse.xml";

    if (lib == Xpetra::UseTpetra) {
#if defined(HAVE_MUELU_TPETRA) && defined(HAVE_MUELU_TPETRA_INST_INT_INT)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      typedef Tpetra::Operator<SC,LO,GO,NO> tpetra_operator_type;
#else
      typedef Tpetra::Operator<SC,NO> tpetra_operator_type;
#endif

      // Matrix
      GO nx = 1000;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Matrix>     Op  = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(nx * comm->getSize(), lib);
#else
      RCP<Matrix>     Op  = TestHelpers::TestFactory<SC, NO>::Build1DPoisson(nx * comm->getSize(), lib);
#endif
      RCP<const Map > map = Op->getRowMap();

      // Normalized RHS
      RCP<MultiVector> RHS1 = MultiVectorFactory::Build(Op->getRowMap(), 1);
      RHS1->setSeed(846930886);
      RHS1->randomize();
      Teuchos::Array<typename Teuchos::ScalarTraits<SC>::magnitudeType> norms(1);
      RHS1->norm2(norms);
      RHS1->scale(1/norms[0]);

      // Zero initial guess
      RCP<MultiVector> X1   = MultiVectorFactory::Build(Op->getRowMap(), 1);
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Tpetra::CrsMatrix<SC,LO,GO,NO> > tpA = MueLu::Utilities<SC,LO,GO,NO>::Op2NonConstTpetraCrs(Op);
#else
      RCP<Tpetra::CrsMatrix<SC,NO> > tpA = MueLu::Utilities<SC,NO>::Op2NonConstTpetraCrs(Op);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<MueLu::TpetraOperator<SC,LO,GO,NO> > tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(RCP<tpetra_operator_type>(tpA), xmlFileName);
#else
      RCP<MueLu::TpetraOperator<SC,NO> > tH = MueLu::CreateTpetraPreconditioner<SC,NO>(RCP<tpetra_operator_type>(tpA), xmlFileName);
#endif
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

#ifdef HAVE_MUELU_DEPRECATED_TESTS

      out << "Testing deprecated method" << std::endl;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      tH = MueLu::CreateTpetraPreconditioner<SC,LO,GO,NO>(tpA, xmlFileName);
#else
      tH = MueLu::CreateTpetraPreconditioner<SC,NO>(tpA, xmlFileName);
#endif
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

#endif // HAVE_MUELU_DEPRECATED_TESTS

      // Reuse preconditioner
      MueLu::ReuseTpetraPreconditioner(tpA, *tH);

      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      tH->apply(*(Utils::MV2TpetraMV(RHS1)), *(Utils::MV2NonConstTpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;
#else
      std::cout << "Skip PetraOperator::ReusePreconditioner: Tpetra is not available (with GO=int enabled)" << std::endl;
#endif // #if defined(HAVE_MUELU_TPETRA) && defined(HAVE_MUELU_TPETRA_INST_INT_INT)

    } else if (lib == Xpetra::UseEpetra) {
#ifdef HAVE_MUELU_EPETRA
      // Matrix
      GO nx = 1000;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Matrix>     Op  = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(nx * comm->getSize(), lib);
#else
      RCP<Matrix>     Op  = TestHelpers::TestFactory<SC, NO>::Build1DPoisson(nx * comm->getSize(), lib);
#endif
      RCP<const Map > map = Op->getRowMap();

      // Normalized RHS
      RCP<MultiVector> RHS1 = MultiVectorFactory::Build(Op->getRowMap(), 1);
      RHS1->setSeed(846930886);
      RHS1->randomize();
      Teuchos::Array<typename Teuchos::ScalarTraits<SC>::magnitudeType> norms(1);
      RHS1->norm2(norms);
      RHS1->scale(1/norms[0]);

      // Zero initial guess
      RCP<MultiVector> X1   = MultiVectorFactory::Build(Op->getRowMap(), 1);
      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<Epetra_CrsMatrix> epA = MueLu::Utilities<SC,LO,GO,NO>::Op2NonConstEpetraCrs(Op);
#else
      RCP<Epetra_CrsMatrix> epA = MueLu::Utilities<SC,NO>::Op2NonConstEpetraCrs(Op);
#endif

      Teuchos::ParameterList paramList;
      Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName,
                                                       Teuchos::Ptr<Teuchos::ParameterList>(&paramList),
                                                       *map->getComm());
      paramList.set("use kokkos refactor", false);
      RCP<MueLu::EpetraOperator> eH = MueLu::CreateEpetraPreconditioner(epA, paramList);

      eH->Apply(*(Utils::MV2EpetraMV(RHS1)), *(Utils::MV2NonConstEpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;

      // Reuse preconditioner
      MueLu::ReuseEpetraPreconditioner(epA, *eH);

      X1->putScalar(Teuchos::ScalarTraits<SC>::zero());
      eH->Apply(*(Utils::MV2EpetraMV(RHS1)), *(Utils::MV2NonConstEpetraMV(X1)));
      out << "after apply, ||b-A*x||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
          Utils::ResidualNorm(*Op, *X1, *RHS1) << std::endl;
#else
      std::cout << "Skip PetraOperator::ReusePreconditioner: Epetra is not available" << std::endl;
#endif

    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, MueLu::Exceptions::InvalidArgument, "Unknown Xpetra lib");
    }
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
#  define MUELU_ETI_GROUP(Scalar, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(PetraOperator, CreatePreconditioner, Scalar, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(PetraOperator, CreatePreconditioner_XMLOnList, Scalar, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(PetraOperator, CreatePreconditioner_PDESystem, Scalar, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(PetraOperator, ReusePreconditioner, Scalar, LO, GO, Node) \
#else
#  define MUELU_ETI_GROUP(Scalar, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(PetraOperator, CreatePreconditioner, Scalar, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(PetraOperator, CreatePreconditioner_XMLOnList, Scalar, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(PetraOperator, CreatePreconditioner_PDESystem, Scalar, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(PetraOperator, ReusePreconditioner, Scalar, Node) \
#endif

#include <MueLu_ETI_4arg.hpp>

}//namespace MueLuTests

// #endif
