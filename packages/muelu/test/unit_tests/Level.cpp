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

#include <MueLu_ConfigDefs.hpp>
#include <MueLu_Version.hpp>

#include <MueLu_Utilities.hpp>

#include <MueLu_NoFactory.hpp>
#include <MueLu_Factory.hpp>

#include <MueLu_TestHelpers.hpp>

#include <MueLu_Level.hpp>
#include <MueLu_NullspaceFactory.hpp>
#include <MueLu_CoalesceDropFactory.hpp>
#include <MueLu_CoupledAggregationFactory.hpp>

#include <MueLu_SingleLevelFactoryBase.hpp>
#include <MueLu_Factory.hpp>

namespace MueLuTests {

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, SetCoreData, Scalar, LocalOrdinal, GlobalOrdinal, Node)
#else
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, SetCoreData, Scalar, Node)
#endif
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    out << "version: " << MueLu::Version() << std::endl;

    Level aLevel;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TestHelpers::TestFactory<SC, LO, GO, NO>::createSingleLevelHierarchy(aLevel);
#else
    TestHelpers::TestFactory<SC, NO>::createSingleLevelHierarchy(aLevel);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Matrix> A = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(2); //can be an empty operator
#else
    RCP<Matrix> A = TestHelpers::TestFactory<SC, NO>::Build1DPoisson(2); //can be an empty operator
#endif

    aLevel.Set("Hitchhiker's Guide", 42);
    int fff = aLevel.Get<int>("Hitchhiker's Guide");
    TEST_EQUALITY(fff, 42);

    aLevel.Set("PI",3.14159265);
    double ggg = aLevel.Get<double>("PI");
    TEST_EQUALITY(ggg, 3.14159265);
    TEST_EQUALITY(aLevel.IsAvailable("PI"), true);

    aLevel.Delete("PI", MueLu::NoFactory::get());
    TEST_EQUALITY(aLevel.IsAvailable("PI"), false);

    aLevel.Set("Hello MueLu", std::string("Greetings to MueMat"));
    std::string hhh = aLevel.Get<std::string>("Hello MueLu");
    TEST_EQUALITY(hhh, "Greetings to MueMat");

    aLevel.Set("A",A);
    RCP<Matrix> newA = aLevel.Get< RCP<Matrix> >("A");
    TEST_EQUALITY(newA, A);

    aLevel.Set("R", A);
    RCP<Matrix> newR = aLevel.Get< RCP<Matrix> >("R");
    TEST_EQUALITY(newR, A); //TODO from JG: must be tested using another matrix !

    aLevel.Set("P", A);
    RCP<Matrix> newP = aLevel.Get< RCP<Matrix> >("P");
    TEST_EQUALITY(newP, A);

    aLevel.SetLevelID(42);
    TEST_EQUALITY(aLevel.GetLevelID(), 42); //TODO: test default value of LevelID
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, NumRequests, Scalar, LocalOrdinal, GlobalOrdinal, Node)
#else
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, NumRequests, Scalar, Node)
#endif
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    out << "version: " << MueLu::Version() << std::endl;

    Level aLevel;
    aLevel.SetLevelID(0);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Matrix> A = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(2);
#else
    RCP<Matrix> A = TestHelpers::TestFactory<SC, NO>::Build1DPoisson(2);
#endif
    aLevel.Set("A", A);

    RCP<FactoryManager> facManager = rcp(new FactoryManager());
    aLevel.SetFactoryManager(facManager);
    RCP<FactoryBase> factory = rcp(new CoalesceDropFactory());

    aLevel.Request("Graph", factory.get());
    aLevel.Request("Graph", factory.get());

    aLevel.Release("Graph", factory.get());
    TEST_EQUALITY(aLevel.IsRequested("Graph", factory.get()), true);

    aLevel.Release("Graph", factory.get());
    TEST_EQUALITY(aLevel.IsRequested("Graph", factory.get()), false);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, RequestRelease, Scalar, LocalOrdinal, GlobalOrdinal, Node)
#else
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, RequestRelease, Scalar, Node)
#endif
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    Level l;
    l.SetLevelID(0);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Matrix> A = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(2);
#else
    RCP<Matrix> A = TestHelpers::TestFactory<SC, NO>::Build1DPoisson(2);
#endif
    l.Set("A", A);

    RCP<FactoryManager> facManager = rcp(new FactoryManager());
    l.SetFactoryManager(facManager);

    RCP<FactoryBase> factory = rcp(new CoalesceDropFactory());

    l.Request("Graph", factory.get());
    TEST_EQUALITY(l.IsRequested("Graph", factory.get()), true);
    TEST_EQUALITY(l.IsAvailable("Graph", factory.get()), false);
    l.Release("Graph", factory.get());
    TEST_EQUALITY(l.IsRequested("Graph", factory.get()), false);
    TEST_EQUALITY(l.IsAvailable("Graph", factory.get()), false);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, RequestReleaseFactory, Scalar, LocalOrdinal, GlobalOrdinal, Node)
#else
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, RequestReleaseFactory, Scalar, Node)
#endif
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    Level l;
    l.SetLevelID(0);

    RCP<FactoryManager> facManager = rcp(new FactoryManager());
    l.SetFactoryManager(facManager);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Matrix> A = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(2);
#else
    RCP<Matrix> A = TestHelpers::TestFactory<SC, NO>::Build1DPoisson(2);
#endif
    l.Set("A", A);

    RCP<FactoryBase> graphFact = rcp(new CoalesceDropFactory());
    RCP<Factory> aggFact  = rcp(new CoupledAggregationFactory());
    aggFact->SetFactory("Graph", graphFact);

    l.Request("Aggregates", aggFact.get());
    TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   true);
    TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   false);

    TEST_EQUALITY(l.IsRequested("Graph",      graphFact.get()), true);
    TEST_EQUALITY(l.IsAvailable("Graph",      graphFact.get()), false);

    l.Release("Aggregates", aggFact.get());
    TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   false);
    TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   false);

    TEST_EQUALITY(l.IsRequested("Graph",      graphFact.get()), false);
    TEST_EQUALITY(l.IsAvailable("Graph",      graphFact.get()), false);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, KeepFactory, Scalar, LocalOrdinal, GlobalOrdinal, Node)
#else
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, KeepFactory, Scalar, Node)
#endif
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    Level l;
    l.SetLevelID(0);

    RCP<FactoryManager> facManager = rcp(new FactoryManager());
    l.SetFactoryManager(facManager);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Matrix> A = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(2);
#else
    RCP<Matrix> A = TestHelpers::TestFactory<SC, NO>::Build1DPoisson(2);
#endif
    l.Set("A", A);

    RCP<Factory> graphFact = rcp(new CoalesceDropFactory());
    RCP<Factory> aggFact   = rcp(new CoupledAggregationFactory());
    aggFact->SetFactory("Graph", graphFact);

    l.Keep("Aggregates", aggFact.get());      // set keep flag
    TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   false);
    TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   false);
    TEST_EQUALITY(l.GetKeepFlag("Aggregates", aggFact.get()),   MueLu::Keep);
    l.Request("Aggregates", aggFact.get());
    TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   true);
    TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   false);
    TEST_EQUALITY(l.GetKeepFlag("Aggregates", aggFact.get()),   MueLu::Keep);

    TEST_EQUALITY(l.IsRequested("Graph",      graphFact.get()), true);
    TEST_EQUALITY(l.IsAvailable("Graph",      graphFact.get()), false);
    TEST_EQUALITY(l.GetKeepFlag("Graph",      graphFact.get()), 0);

    l.Release("Aggregates", aggFact.get());
    TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   false);
    TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   false);
    TEST_EQUALITY(l.GetKeepFlag("Aggregates", aggFact.get()),   MueLu::Keep);

    TEST_EQUALITY(l.IsRequested("Graph",      graphFact.get()), false);
    TEST_EQUALITY(l.IsAvailable("Graph",      graphFact.get()), false);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, KeepAndBuildFactory, Scalar, LocalOrdinal, GlobalOrdinal, Node)
#else
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, KeepAndBuildFactory, Scalar, Node)
#endif
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    Level l;
    l.SetLevelID(0); // level 0 necessary because of Nullspace factory

    RCP<FactoryManager> facManager = rcp(new FactoryManager());
    l.SetFactoryManager(facManager);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Matrix> A = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(144);
#else
    RCP<Matrix> A = TestHelpers::TestFactory<SC, NO>::Build1DPoisson(144);
#endif
    l.Set("A", A);

    RCP<CoalesceDropFactory>  graphFact = rcp(new CoalesceDropFactory());
    RCP<CoupledAggregationFactory> aggFact   = rcp(new CoupledAggregationFactory());
    aggFact->SetFactory("Graph", graphFact);

    l.Keep("Aggregates", aggFact.get());      // set keep flag
    TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   false);
    TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   false);
    TEST_EQUALITY(l.GetKeepFlag("Aggregates", aggFact.get()),   MueLu::Keep);
    l.Request("Aggregates", aggFact.get());
    TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   true);
    TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   false);
    TEST_EQUALITY(l.GetKeepFlag("Aggregates", aggFact.get()),   MueLu::Keep);

    aggFact->Build(l);

    TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   true);
    TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   true);
    TEST_EQUALITY(l.GetKeepFlag("Aggregates", aggFact.get()),   MueLu::Keep);

    TEST_EQUALITY(l.IsRequested("Graph",      graphFact.get()), true);
    TEST_EQUALITY(l.IsAvailable("Graph",      graphFact.get()), true);
    TEST_EQUALITY(l.GetKeepFlag("Graph",      graphFact.get()), 0);

    l.Release(*aggFact); // release dependencies only

    TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   true);
    TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   true);
    TEST_EQUALITY(l.GetKeepFlag("Aggregates", aggFact.get()),   MueLu::Keep);

    TEST_EQUALITY(l.IsRequested("Graph",      graphFact.get()), false);
    TEST_EQUALITY(l.IsAvailable("Graph",      graphFact.get()), false);
    TEST_EQUALITY(l.GetKeepFlag("Graph",      graphFact.get()), 0);

    l.Release("Aggregates", aggFact.get());

    TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   false);
    TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   true);
    TEST_EQUALITY(l.GetKeepFlag("Aggregates", aggFact.get()),   MueLu::Keep);

    TEST_EQUALITY(l.IsRequested("Graph",      graphFact.get()), false);
    TEST_EQUALITY(l.IsAvailable("Graph",      graphFact.get()), false);
    TEST_EQUALITY(l.GetKeepFlag("Graph",      graphFact.get()), 0);

    l.RemoveKeepFlag("Aggregates", aggFact.get(), MueLu::Keep);

    TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   false);
    TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   false);

    TEST_EQUALITY(l.IsRequested("Graph",      graphFact.get()), false);
    TEST_EQUALITY(l.IsAvailable("Graph",      graphFact.get()), false);
    TEST_EQUALITY(l.GetKeepFlag("Graph",      graphFact.get()), 0);

  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, KeepAndBuildFactory2, Scalar, LocalOrdinal, GlobalOrdinal, Node)
#else
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, KeepAndBuildFactory2, Scalar, Node)
#endif
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    Level l;
    l.SetLevelID(0); // level 0 necessary because of Nullspace factory

    RCP<FactoryManager> facManager = rcp(new FactoryManager());
    l.SetFactoryManager(facManager);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Matrix> A = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(144);
#else
    RCP<Matrix> A = TestHelpers::TestFactory<SC, NO>::Build1DPoisson(144);
#endif
    l.Set("A", A);

    RCP<CoalesceDropFactory>  graphFact = rcp(new CoalesceDropFactory());
    RCP<CoupledAggregationFactory> aggFact   = rcp(new CoupledAggregationFactory());
    aggFact->SetFactory("Graph", graphFact);

    TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   false);
    TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   false);

    l.Request("Aggregates", aggFact.get());
    TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   true);
    TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   false);

    aggFact->Build(l);

    TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   true);
    TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   true);

    TEST_EQUALITY(l.IsRequested("Graph",      graphFact.get()), true);
    TEST_EQUALITY(l.IsAvailable("Graph",      graphFact.get()), true);
    TEST_EQUALITY(l.GetKeepFlag("Graph",      graphFact.get()), 0);

    l.Release(*aggFact);

    TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   true);
    TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   true);

    TEST_EQUALITY(l.IsRequested("Graph",      graphFact.get()), false);
    TEST_EQUALITY(l.IsAvailable("Graph",      graphFact.get()), false);
    TEST_EQUALITY(l.GetKeepFlag("Graph",      graphFact.get()), 0);

    l.Release("Aggregates", aggFact.get());

    TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   false);
    TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   false);

    TEST_EQUALITY(l.IsRequested("Graph",      graphFact.get()), false);
    TEST_EQUALITY(l.IsAvailable("Graph",      graphFact.get()), false);
    TEST_EQUALITY(l.GetKeepFlag("Graph",      graphFact.get()), 0);

    /*l.RemoveKeepFlag("Aggregates", aggFact.get(), MueLu::Keep);

      TEST_EQUALITY(l.IsRequested("Aggregates", aggFact.get()),   false);
      TEST_EQUALITY(l.IsAvailable("Aggregates", aggFact.get()),   false);

      TEST_EQUALITY(l.IsRequested("Graph",      graphFact.get()), false);
      TEST_EQUALITY(l.IsAvailable("Graph",      graphFact.get()), false);
      TEST_EQUALITY(l.GetKeepFlag("Graph",      graphFact.get()), 0);*/

  }


  // Helper class for unit test 'Level/CircularDependency'
  template
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
  <class Scalar, class Node>
#endif
  class CircularFactory : public MueLu::SingleLevelFactoryBase {

#   include <MueLu_UseShortNames.hpp>

    public:

#ifndef TPETRA_ENABLE_TEMPLATE_ORDINALS
      using LocalOrdinal = typename Tpetra::Map<>::local_ordinal_type;
      using GlobalOrdinal = typename Tpetra::Map<>::global_ordinal_type;
#endif
      CircularFactory(int value) : value_(value) { }

      virtual ~CircularFactory() { }

      void SetCircularFactory(RCP<FactoryBase> circular) { circular_ = circular; }

      void DeclareInput(Level &level) const {
        level.DeclareInput("data", circular_.get(), this);
      }

      void Build(Level& level) const {
        level.Set("data", value_, this);
        int value = level.Get<int>("data", circular_.get());
        level.Set("data", value+value_, this);
      }

    private:

      int value_;
      RCP<FactoryBase> circular_;

  }; //class CircularFactory

  //! Even though it's very special, a factory can generate data, that it requests itself.
  //  Level must avoid self-recursive calls of Request
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, CircularDependencyWith1Factory, Scalar, LocalOrdinal, GlobalOrdinal, Node)
#else
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, CircularDependencyWith1Factory, Scalar, Node)
#endif
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef CircularFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node> circular_factory_type;
#else
    typedef CircularFactory<Scalar, Node> circular_factory_type;
#endif
    circular_factory_type A(2);

    A.SetCircularFactory(rcpFromRef(A));

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Level level; TestHelpers::TestFactory<SC, LO, GO, NO>::createSingleLevelHierarchy(level);
#else
    Level level; TestHelpers::TestFactory<SC, NO>::createSingleLevelHierarchy(level);
#endif

    level.Request("data", &A);

    TEST_EQUALITY(level.Get<int>("data", &A), (2 + 2));

    level.Release("data", &A);
  }

  //! Test if circular dependencies between factories are allowed
  //  This test corresponds to a use-case found developping repartitionning capability
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, CircularDependencyWithTwoFactories, Scalar, LocalOrdinal, GlobalOrdinal, Node)
#else
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Level, CircularDependencyWithTwoFactories, Scalar, Node)
#endif
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef CircularFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node> circular_factory_type;
#else
    typedef CircularFactory<Scalar, Node> circular_factory_type;
#endif
    circular_factory_type A(2);
    circular_factory_type B(3);

    A.SetCircularFactory(rcpFromRef(B));
    B.SetCircularFactory(rcpFromRef(A));

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Level level; TestHelpers::TestFactory<SC, LO, GO, NO>::createSingleLevelHierarchy(level);
#else
    Level level; TestHelpers::TestFactory<SC, NO>::createSingleLevelHierarchy(level);
#endif

    level.Request("data", &A);

    A.Build(level);

    TEST_EQUALITY(level.Get<int>("data", &A), (2 + 3) + 2);
    TEST_EQUALITY(level.Get<int>("data", &B), (2 + 3));

    level.Release(A); // needed because A.Build(level) have been called manually
    level.Release("data", &A);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
   #define MUELU_ETI_GROUP(SC,LO,GO,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,SetCoreData,SC,LO,GO,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,NumRequests,SC,LO,GO,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,RequestRelease,SC,LO,GO,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,RequestReleaseFactory,SC,LO,GO,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,KeepFactory,SC,LO,GO,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,KeepAndBuildFactory,SC,LO,GO,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,KeepAndBuildFactory2,SC,LO,GO,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,CircularDependencyWith1Factory,SC,LO,GO,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,CircularDependencyWithTwoFactories,SC,LO,GO,NO)
#else
   #define MUELU_ETI_GROUP(SC,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,SetCoreData,SC,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,NumRequests,SC,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,RequestRelease,SC,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,RequestReleaseFactory,SC,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,KeepFactory,SC,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,KeepAndBuildFactory,SC,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,KeepAndBuildFactory2,SC,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,CircularDependencyWith1Factory,SC,NO) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Level,CircularDependencyWithTwoFactories,SC,NO)
#endif

} // namespace MueLuTests

