// Type definitions for templated classes (generally graph-related) that do not require a scalar.

#include <Xpetra_UseShortNamesOrdinal.hpp>

#ifdef MUELU_AGGREGATES_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::Aggregates<LocalOrdinal,GlobalOrdinal,Node> Aggregates;
#else
typedef MueLu::Aggregates<Node> Aggregates;
#endif
#endif
#ifdef MUELU_AGGREGATES_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::Aggregates_kokkos<LocalOrdinal,GlobalOrdinal,Node> Aggregates_kokkos;
#else
typedef MueLu::Aggregates_kokkos<Node> Aggregates_kokkos;
#endif
#endif
#ifdef MUELU_AGGREGATIONPHASE1ALGORITHM_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AggregationPhase1Algorithm<LocalOrdinal,GlobalOrdinal,Node> AggregationPhase1Algorithm;
#else
typedef MueLu::AggregationPhase1Algorithm<Node> AggregationPhase1Algorithm;
#endif
#endif
#ifdef MUELU_AGGREGATIONPHASE1ALGORITHM_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AggregationPhase1Algorithm_kokkos<LocalOrdinal,GlobalOrdinal,Node> AggregationPhase1Algorithm_kokkos;
#else
typedef MueLu::AggregationPhase1Algorithm_kokkos<Node> AggregationPhase1Algorithm_kokkos;
#endif
#endif
#ifdef MUELU_AGGREGATIONPHASE2AALGORITHM_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AggregationPhase2aAlgorithm<LocalOrdinal,GlobalOrdinal,Node> AggregationPhase2aAlgorithm;
#else
typedef MueLu::AggregationPhase2aAlgorithm<Node> AggregationPhase2aAlgorithm;
#endif
#endif
#ifdef MUELU_AGGREGATIONPHASE2AALGORITHM_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AggregationPhase2aAlgorithm_kokkos<LocalOrdinal,GlobalOrdinal,Node> AggregationPhase2aAlgorithm_kokkos;
#else
typedef MueLu::AggregationPhase2aAlgorithm_kokkos<Node> AggregationPhase2aAlgorithm_kokkos;
#endif
#endif
#ifdef MUELU_AGGREGATIONPHASE2BALGORITHM_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AggregationPhase2bAlgorithm<LocalOrdinal,GlobalOrdinal,Node> AggregationPhase2bAlgorithm;
#else
typedef MueLu::AggregationPhase2bAlgorithm<Node> AggregationPhase2bAlgorithm;
#endif
#endif
#ifdef MUELU_AGGREGATIONPHASE2BALGORITHM_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AggregationPhase2bAlgorithm_kokkos<LocalOrdinal,GlobalOrdinal,Node> AggregationPhase2bAlgorithm_kokkos;
#else
typedef MueLu::AggregationPhase2bAlgorithm_kokkos<Node> AggregationPhase2bAlgorithm_kokkos;
#endif
#endif
#ifdef MUELU_AGGREGATIONPHASE3ALGORITHM_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AggregationPhase3Algorithm<LocalOrdinal,GlobalOrdinal,Node> AggregationPhase3Algorithm;
#else
typedef MueLu::AggregationPhase3Algorithm<Node> AggregationPhase3Algorithm;
#endif
#endif
#ifdef MUELU_AGGREGATIONPHASE3ALGORITHM_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AggregationPhase3Algorithm_kokkos<LocalOrdinal,GlobalOrdinal,Node> AggregationPhase3Algorithm_kokkos;
#else
typedef MueLu::AggregationPhase3Algorithm_kokkos<Node> AggregationPhase3Algorithm_kokkos;
#endif
#endif
#ifdef MUELU_AGGREGATIONSTRUCTUREDALGORITHM_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AggregationStructuredAlgorithm<LocalOrdinal,GlobalOrdinal,Node> AggregationStructuredAlgorithm;
#else
typedef MueLu::AggregationStructuredAlgorithm<Node> AggregationStructuredAlgorithm;
#endif
#endif
#ifdef MUELU_AGGREGATIONSTRUCTUREDALGORITHM_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AggregationStructuredAlgorithm_kokkos<LocalOrdinal,GlobalOrdinal,Node> AggregationStructuredAlgorithm_kokkos;
#else
typedef MueLu::AggregationStructuredAlgorithm_kokkos<Node> AggregationStructuredAlgorithm_kokkos;
#endif
#endif
#ifdef MUELU_AMALGAMATIONINFO_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AmalgamationInfo<LocalOrdinal,GlobalOrdinal,Node> AmalgamationInfo;
#else
typedef MueLu::AmalgamationInfo<Node> AmalgamationInfo;
#endif
#endif
#ifdef MUELU_AMALGAMATIONINFO_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AmalgamationInfo_kokkos<LocalOrdinal,GlobalOrdinal,Node> AmalgamationInfo_kokkos;
#else
typedef MueLu::AmalgamationInfo_kokkos<Node> AmalgamationInfo_kokkos;
#endif
#endif
#ifdef MUELU_COUPLEDAGGREGATIONCOMMHELPER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::CoupledAggregationCommHelper<LocalOrdinal,GlobalOrdinal,Node> CoupledAggregationCommHelper;
#else
typedef MueLu::CoupledAggregationCommHelper<Node> CoupledAggregationCommHelper;
#endif
#endif
#ifdef MUELU_COUPLEDAGGREGATIONFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::CoupledAggregationFactory<LocalOrdinal,GlobalOrdinal,Node> CoupledAggregationFactory;
#else
typedef MueLu::CoupledAggregationFactory<Node> CoupledAggregationFactory;
#endif
#endif
#ifdef MUELU_GLOBALLEXICOGRAPHICINDEXMANAGER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::GlobalLexicographicIndexManager<LocalOrdinal,GlobalOrdinal,Node> GlobalLexicographicIndexManager;
#else
typedef MueLu::GlobalLexicographicIndexManager<Node> GlobalLexicographicIndexManager;
#endif
#endif
#ifdef MUELU_GRAPH_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::Graph<LocalOrdinal,GlobalOrdinal,Node> Graph;
#else
typedef MueLu::Graph<Node> Graph;
#endif
#endif
#ifdef MUELU_GRAPHBASE_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::GraphBase<LocalOrdinal,GlobalOrdinal,Node> GraphBase;
#else
typedef MueLu::GraphBase<Node> GraphBase;
#endif
#endif
#ifdef MUELU_HYBRIDAGGREGATIONFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::HybridAggregationFactory<LocalOrdinal,GlobalOrdinal,Node> HybridAggregationFactory;
#else
typedef MueLu::HybridAggregationFactory<Node> HybridAggregationFactory;
#endif
#endif
#ifdef MUELU_INDEXMANAGER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::IndexManager<LocalOrdinal,GlobalOrdinal,Node> IndexManager;
#else
typedef MueLu::IndexManager<Node> IndexManager;
#endif
#endif
#ifdef MUELU_INDEXMANAGER_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::IndexManager_kokkos<LocalOrdinal,GlobalOrdinal,Node> IndexManager_kokkos;
#else
typedef MueLu::IndexManager_kokkos<Node> IndexManager_kokkos;
#endif
#endif
#ifdef MUELU_INTERFACEAGGREGATIONALGORITHM_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::InterfaceAggregationAlgorithm<LocalOrdinal,GlobalOrdinal,Node> InterfaceAggregationAlgorithm;
#else
typedef MueLu::InterfaceAggregationAlgorithm<Node> InterfaceAggregationAlgorithm;
#endif
#endif
#ifdef MUELU_ISOLATEDNODEAGGREGATIONALGORITHM_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::IsolatedNodeAggregationAlgorithm<LocalOrdinal,GlobalOrdinal,Node> IsolatedNodeAggregationAlgorithm;
#else
typedef MueLu::IsolatedNodeAggregationAlgorithm<Node> IsolatedNodeAggregationAlgorithm;
#endif
#endif
#ifdef MUELU_ISOLATEDNODEAGGREGATIONALGORITHM_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::IsolatedNodeAggregationAlgorithm_kokkos<LocalOrdinal,GlobalOrdinal,Node> IsolatedNodeAggregationAlgorithm_kokkos;
#else
typedef MueLu::IsolatedNodeAggregationAlgorithm_kokkos<Node> IsolatedNodeAggregationAlgorithm_kokkos;
#endif
#endif
#ifdef MUELU_ISORROPIAINTERFACE_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::IsorropiaInterface<LocalOrdinal,GlobalOrdinal,Node> IsorropiaInterface;
#else
typedef MueLu::IsorropiaInterface<Node> IsorropiaInterface;
#endif
#endif
#ifdef MUELU_LWGRAPH_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::LWGraph<LocalOrdinal,GlobalOrdinal,Node> LWGraph;
#else
typedef MueLu::LWGraph<Node> LWGraph;
#endif
#endif
#ifdef MUELU_LWGRAPH_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::LWGraph_kokkos<LocalOrdinal,GlobalOrdinal,Node> LWGraph_kokkos;
#else
typedef MueLu::LWGraph_kokkos<Node> LWGraph_kokkos;
#endif
#endif
#ifdef MUELU_LEFTOVERAGGREGATIONALGORITHM_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::LeftoverAggregationAlgorithm<LocalOrdinal,GlobalOrdinal,Node> LeftoverAggregationAlgorithm;
#else
typedef MueLu::LeftoverAggregationAlgorithm<Node> LeftoverAggregationAlgorithm;
#endif
#endif
#ifdef MUELU_LOCALAGGREGATIONALGORITHM_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::LocalAggregationAlgorithm<LocalOrdinal,GlobalOrdinal,Node> LocalAggregationAlgorithm;
#else
typedef MueLu::LocalAggregationAlgorithm<Node> LocalAggregationAlgorithm;
#endif
#endif
#ifdef MUELU_LOCALLEXICOGRAPHICINDEXMANAGER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::LocalLexicographicIndexManager<LocalOrdinal,GlobalOrdinal,Node> LocalLexicographicIndexManager;
#else
typedef MueLu::LocalLexicographicIndexManager<Node> LocalLexicographicIndexManager;
#endif
#endif
#ifdef MUELU_ONEPTAGGREGATIONALGORITHM_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::OnePtAggregationAlgorithm<LocalOrdinal,GlobalOrdinal,Node> OnePtAggregationAlgorithm;
#else
typedef MueLu::OnePtAggregationAlgorithm<Node> OnePtAggregationAlgorithm;
#endif
#endif
#ifdef MUELU_ONEPTAGGREGATIONALGORITHM_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::OnePtAggregationAlgorithm_kokkos<LocalOrdinal,GlobalOrdinal,Node> OnePtAggregationAlgorithm_kokkos;
#else
typedef MueLu::OnePtAggregationAlgorithm_kokkos<Node> OnePtAggregationAlgorithm_kokkos;
#endif
#endif
#ifdef MUELU_PRESERVEDIRICHLETAGGREGATIONALGORITHM_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::PreserveDirichletAggregationAlgorithm<LocalOrdinal,GlobalOrdinal,Node> PreserveDirichletAggregationAlgorithm;
#else
typedef MueLu::PreserveDirichletAggregationAlgorithm<Node> PreserveDirichletAggregationAlgorithm;
#endif
#endif
#ifdef MUELU_PRESERVEDIRICHLETAGGREGATIONALGORITHM_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::PreserveDirichletAggregationAlgorithm_kokkos<LocalOrdinal,GlobalOrdinal,Node> PreserveDirichletAggregationAlgorithm_kokkos;
#else
typedef MueLu::PreserveDirichletAggregationAlgorithm_kokkos<Node> PreserveDirichletAggregationAlgorithm_kokkos;
#endif
#endif
#ifdef MUELU_PRFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::PRFactory<LocalOrdinal,GlobalOrdinal,Node> PRFactory;
#else
typedef MueLu::PRFactory<Node> PRFactory;
#endif
#endif
#ifdef MUELU_REBALANCEMAPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::RebalanceMapFactory<LocalOrdinal,GlobalOrdinal,Node> RebalanceMapFactory;
#else
typedef MueLu::RebalanceMapFactory<Node> RebalanceMapFactory;
#endif
#endif
#ifdef MUELU_REPARTITIONINTERFACE_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::RepartitionInterface<LocalOrdinal,GlobalOrdinal,Node> RepartitionInterface;
#else
typedef MueLu::RepartitionInterface<Node> RepartitionInterface;
#endif
#endif
#ifdef MUELU_STRUCTUREDAGGREGATIONFACTORY_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::StructuredAggregationFactory_kokkos<LocalOrdinal,GlobalOrdinal,Node> StructuredAggregationFactory_kokkos;
#else
typedef MueLu::StructuredAggregationFactory_kokkos<Node> StructuredAggregationFactory_kokkos;
#endif
#endif
#ifdef MUELU_UNCOUPLEDAGGREGATIONFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::UncoupledAggregationFactory<LocalOrdinal,GlobalOrdinal,Node> UncoupledAggregationFactory;
#else
typedef MueLu::UncoupledAggregationFactory<Node> UncoupledAggregationFactory;
#endif
#endif
#ifdef MUELU_UNCOUPLEDAGGREGATIONFACTORY_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::UncoupledAggregationFactory_kokkos<LocalOrdinal,GlobalOrdinal,Node> UncoupledAggregationFactory_kokkos;
#else
typedef MueLu::UncoupledAggregationFactory_kokkos<Node> UncoupledAggregationFactory_kokkos;
#endif
#endif
#ifdef MUELU_UNCOUPLEDINDEXMANAGER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::UncoupledIndexManager<LocalOrdinal,GlobalOrdinal,Node> UncoupledIndexManager;
#else
typedef MueLu::UncoupledIndexManager<Node> UncoupledIndexManager;
#endif
#endif
#ifdef MUELU_USERAGGREGATIONFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::UserAggregationFactory<LocalOrdinal,GlobalOrdinal,Node> UserAggregationFactory;
#else
typedef MueLu::UserAggregationFactory<Node> UserAggregationFactory;
#endif
#endif
#ifdef MUELU_FACTORY_SHORT
typedef MueLu::Factory Factory;
#endif

#ifdef MUELU_FACTORYBASE_SHORT
typedef MueLu::FactoryBase FactoryBase;
#endif

#ifdef MUELU_FACTORYMANAGERBASE_SHORT
typedef MueLu::FactoryManagerBase FactoryManagerBase;
#endif

#ifdef MUELU_LEVEL_SHORT
typedef MueLu::Level Level;
#endif

#ifdef MUELU_PFACTORY_SHORT
typedef MueLu::PFactory PFactory;
#endif

#ifdef MUELU_RFACTORY_SHORT
typedef MueLu::RFactory RFactory;
#endif

#ifdef MUELU_SINGLELEVELFACTORYBASE_SHORT
typedef MueLu::SingleLevelFactoryBase SingleLevelFactoryBase;
#endif

#ifdef MUELU_TWOLEVELFACTORYBASE_SHORT
typedef MueLu::TwoLevelFactoryBase TwoLevelFactoryBase;
#endif

#ifdef MUELU_VARIABLECONTAINER_SHORT
typedef MueLu::VariableContainer VariableContainer;
#endif

#ifdef MUELU_SMOOTHERFACTORYBASE_SHORT
typedef MueLu::SmootherFactoryBase SmootherFactoryBase;
#endif

#ifdef MUELU_AMESOSSMOOTHER_SHORT
typedef MueLu::AmesosSmoother<Node> AmesosSmoother;
#endif
#ifdef MUELU_IFPACKSMOOTHER_SHORT
typedef MueLu::IfpackSmoother<Node> IfpackSmoother;
#endif
