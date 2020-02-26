// New definition of types using the types Scalar, LocalOrdinal, GlobalOrdinal, Node of the current context.

#include <Xpetra_UseShortNamesScalar.hpp>

#ifdef MUELU_AGGREGATIONEXPORTFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AggregationExportFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> AggregationExportFactory;
#else
typedef MueLu::AggregationExportFactory<Scalar,Node> AggregationExportFactory;
#endif
#endif
#ifdef MUELU_AGGREGATEQUALITYESTIMATEFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AggregateQualityEstimateFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> AggregateQualityEstimateFactory;
#else
typedef MueLu::AggregateQualityEstimateFactory<Scalar,Node> AggregateQualityEstimateFactory;
#endif
#endif
#ifdef MUELU_AMALGAMATIONFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AmalgamationFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> AmalgamationFactory;
#else
typedef MueLu::AmalgamationFactory<Scalar,Node> AmalgamationFactory;
#endif
#endif
#ifdef MUELU_AMALGAMATIONFACTORY_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AmalgamationFactory_kokkos<Scalar,LocalOrdinal,GlobalOrdinal,Node> AmalgamationFactory_kokkos;
#else
typedef MueLu::AmalgamationFactory_kokkos<Scalar,Node> AmalgamationFactory_kokkos;
#endif
#endif
#ifdef MUELU_AMESOS2SMOOTHER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::Amesos2Smoother<Scalar,LocalOrdinal,GlobalOrdinal,Node> Amesos2Smoother;
#else
typedef MueLu::Amesos2Smoother<Scalar,Node> Amesos2Smoother;
#endif
#endif
#ifdef MUELU_AMGXOPERATOR_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AMGXOperator<Scalar,LocalOrdinal,GlobalOrdinal,Node> AMGXOperator;
#else
typedef MueLu::AMGXOperator<Scalar,Node> AMGXOperator;
#endif
#endif
#ifdef MUELU_ALGEBRAICPERMUTATIONSTRATEGY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AlgebraicPermutationStrategy<Scalar,LocalOrdinal,GlobalOrdinal,Node> AlgebraicPermutationStrategy;
#else
typedef MueLu::AlgebraicPermutationStrategy<Scalar,Node> AlgebraicPermutationStrategy;
#endif
#endif
#ifdef MUELU_BLACKBOXPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::BlackBoxPFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> BlackBoxPFactory;
#else
typedef MueLu::BlackBoxPFactory<Scalar,Node> BlackBoxPFactory;
#endif
#endif
#ifdef MUELU_BLOCKEDCOARSEMAPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::BlockedCoarseMapFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> BlockedCoarseMapFactory;
#else
typedef MueLu::BlockedCoarseMapFactory<Scalar,Node> BlockedCoarseMapFactory;
#endif
#endif
#ifdef MUELU_BLOCKEDCOORDINATESTRANSFERFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::BlockedCoordinatesTransferFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> BlockedCoordinatesTransferFactory;
#else
typedef MueLu::BlockedCoordinatesTransferFactory<Scalar,Node> BlockedCoordinatesTransferFactory;
#endif
#endif
#ifdef MUELU_BLOCKEDDIRECTSOLVER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::BlockedDirectSolver<Scalar,LocalOrdinal,GlobalOrdinal,Node> BlockedDirectSolver;
#else
typedef MueLu::BlockedDirectSolver<Scalar,Node> BlockedDirectSolver;
#endif
#endif
#ifdef MUELU_BLOCKEDGAUSSSEIDELSMOOTHER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::BlockedGaussSeidelSmoother<Scalar,LocalOrdinal,GlobalOrdinal,Node> BlockedGaussSeidelSmoother;
#else
typedef MueLu::BlockedGaussSeidelSmoother<Scalar,Node> BlockedGaussSeidelSmoother;
#endif
#endif
#ifdef MUELU_BLOCKEDJACOBISMOOTHER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::BlockedJacobiSmoother<Scalar,LocalOrdinal,GlobalOrdinal,Node> BlockedJacobiSmoother;
#else
typedef MueLu::BlockedJacobiSmoother<Scalar,Node> BlockedJacobiSmoother;
#endif
#endif
#ifdef MUELU_BLOCKEDPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::BlockedPFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> BlockedPFactory;
#else
typedef MueLu::BlockedPFactory<Scalar,Node> BlockedPFactory;
#endif
#endif
#ifdef MUELU_BLOCKEDRAPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::BlockedRAPFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> BlockedRAPFactory;
#else
typedef MueLu::BlockedRAPFactory<Scalar,Node> BlockedRAPFactory;
#endif
#endif
#ifdef MUELU_BRICKAGGREGATIONFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::BrickAggregationFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> BrickAggregationFactory;
#else
typedef MueLu::BrickAggregationFactory<Scalar,Node> BrickAggregationFactory;
#endif
#endif
#ifdef MUELU_BRAESSSARAZINSMOOTHER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::BraessSarazinSmoother<Scalar,LocalOrdinal,GlobalOrdinal,Node> BraessSarazinSmoother;
#else
typedef MueLu::BraessSarazinSmoother<Scalar,Node> BraessSarazinSmoother;
#endif
#endif
#ifdef MUELU_CGSOLVER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::CGSolver<Scalar,LocalOrdinal,GlobalOrdinal,Node> CGSolver;
#else
typedef MueLu::CGSolver<Scalar,Node> CGSolver;
#endif
#endif
#ifdef MUELU_CLONEREPARTITIONINTERFACE_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::CloneRepartitionInterface<Scalar,LocalOrdinal,GlobalOrdinal,Node> CloneRepartitionInterface;
#else
typedef MueLu::CloneRepartitionInterface<Scalar,Node> CloneRepartitionInterface;
#endif
#endif
#ifdef MUELU_COALESCEDROPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::CoalesceDropFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> CoalesceDropFactory;
#else
typedef MueLu::CoalesceDropFactory<Scalar,Node> CoalesceDropFactory;
#endif
#endif
#ifdef MUELU_COALESCEDROPFACTORY_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::CoalesceDropFactory_kokkos<Scalar,LocalOrdinal,GlobalOrdinal,Node> CoalesceDropFactory_kokkos;
#else
typedef MueLu::CoalesceDropFactory_kokkos<Scalar,Node> CoalesceDropFactory_kokkos;
#endif
#endif
#ifdef MUELU_COARSEMAPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::CoarseMapFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> CoarseMapFactory;
#else
typedef MueLu::CoarseMapFactory<Scalar,Node> CoarseMapFactory;
#endif
#endif
#ifdef MUELU_COARSEMAPFACTORY_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::CoarseMapFactory_kokkos<Scalar,LocalOrdinal,GlobalOrdinal,Node> CoarseMapFactory_kokkos;
#else
typedef MueLu::CoarseMapFactory_kokkos<Scalar,Node> CoarseMapFactory_kokkos;
#endif
#endif
#ifdef MUELU_COARSENINGVISUALIZATIONFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::CoarseningVisualizationFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> CoarseningVisualizationFactory;
#else
typedef MueLu::CoarseningVisualizationFactory<Scalar,Node> CoarseningVisualizationFactory;
#endif
#endif
#ifdef MUELU_CONSTRAINT_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::Constraint<Scalar,LocalOrdinal,GlobalOrdinal,Node> Constraint;
#else
typedef MueLu::Constraint<Scalar,Node> Constraint;
#endif
#endif
#ifdef MUELU_CONSTRAINTFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::ConstraintFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> ConstraintFactory;
#else
typedef MueLu::ConstraintFactory<Scalar,Node> ConstraintFactory;
#endif
#endif
#ifdef MUELU_COORDINATESTRANSFERFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::CoordinatesTransferFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> CoordinatesTransferFactory;
#else
typedef MueLu::CoordinatesTransferFactory<Scalar,Node> CoordinatesTransferFactory;
#endif
#endif
#ifdef MUELU_COORDINATESTRANSFERFACTORY_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::CoordinatesTransferFactory_kokkos<Scalar,LocalOrdinal,GlobalOrdinal,Node> CoordinatesTransferFactory_kokkos;
#else
typedef MueLu::CoordinatesTransferFactory_kokkos<Scalar,Node> CoordinatesTransferFactory_kokkos;
#endif
#endif
#ifdef MUELU_COUPLEDRBMFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::CoupledRBMFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> CoupledRBMFactory;
#else
typedef MueLu::CoupledRBMFactory<Scalar,Node> CoupledRBMFactory;
#endif
#endif
#ifdef MUELU_DEMOFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::DemoFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> DemoFactory;
#else
typedef MueLu::DemoFactory<Scalar,Node> DemoFactory;
#endif
#endif
#ifdef MUELU_DIRECTSOLVER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::DirectSolver<Scalar,LocalOrdinal,GlobalOrdinal,Node> DirectSolver;
#else
typedef MueLu::DirectSolver<Scalar,Node> DirectSolver;
#endif
#endif
#ifdef MUELU_DROPNEGATIVEENTRIESFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::DropNegativeEntriesFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> DropNegativeEntriesFactory;
#else
typedef MueLu::DropNegativeEntriesFactory<Scalar,Node> DropNegativeEntriesFactory;
#endif
#endif
#ifdef MUELU_EMINPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::EminPFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> EminPFactory;
#else
typedef MueLu::EminPFactory<Scalar,Node> EminPFactory;
#endif
#endif
#ifdef MUELU_FACADECLASSFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::FacadeClassFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> FacadeClassFactory;
#else
typedef MueLu::FacadeClassFactory<Scalar,Node> FacadeClassFactory;
#endif
#endif
#ifdef MUELU_FACTORYMANAGER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::FactoryManager<Scalar,LocalOrdinal,GlobalOrdinal,Node> FactoryManager;
#else
typedef MueLu::FactoryManager<Scalar,Node> FactoryManager;
#endif
#endif
#ifdef MUELU_FAKESMOOTHERPROTOTYPE_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::FakeSmootherPrototype<Scalar,LocalOrdinal,GlobalOrdinal,Node> FakeSmootherPrototype;
#else
typedef MueLu::FakeSmootherPrototype<Scalar,Node> FakeSmootherPrototype;
#endif
#endif
#ifdef MUELU_FILTEREDAFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::FilteredAFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> FilteredAFactory;
#else
typedef MueLu::FilteredAFactory<Scalar,Node> FilteredAFactory;
#endif
#endif
#ifdef MUELU_FINELEVELINPUTDATAFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::FineLevelInputDataFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> FineLevelInputDataFactory;
#else
typedef MueLu::FineLevelInputDataFactory<Scalar,Node> FineLevelInputDataFactory;
#endif
#endif
#ifdef MUELU_GENERALGEOMETRICPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::GeneralGeometricPFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> GeneralGeometricPFactory;
#else
typedef MueLu::GeneralGeometricPFactory<Scalar,Node> GeneralGeometricPFactory;
#endif
#endif
#ifdef MUELU_GENERICRFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::GenericRFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> GenericRFactory;
#else
typedef MueLu::GenericRFactory<Scalar,Node> GenericRFactory;
#endif
#endif
#ifdef MUELU_GEOMETRICINTERPOLATIONPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::GeometricInterpolationPFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> GeometricInterpolationPFactory;
#else
typedef MueLu::GeometricInterpolationPFactory<Scalar,Node> GeometricInterpolationPFactory;
#endif
#endif
#ifdef MUELU_GEOMETRICINTERPOLATIONPFACTORY_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::GeometricInterpolationPFactory_kokkos<Scalar,LocalOrdinal,GlobalOrdinal,Node> GeometricInterpolationPFactory_kokkos;
#else
typedef MueLu::GeometricInterpolationPFactory_kokkos<Scalar,Node> GeometricInterpolationPFactory_kokkos;
#endif
#endif
#ifdef MUELU_GMRESSOLVER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::GMRESSolver<Scalar,LocalOrdinal,GlobalOrdinal,Node> GMRESSolver;
#else
typedef MueLu::GMRESSolver<Scalar,Node> GMRESSolver;
#endif
#endif
#ifdef MUELU_HIERARCHY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::Hierarchy<Scalar,LocalOrdinal,GlobalOrdinal,Node> Hierarchy;
#else
typedef MueLu::Hierarchy<Scalar,Node> Hierarchy;
#endif
#endif
#ifdef MUELU_HIERARCHYMANAGER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::HierarchyManager<Scalar,LocalOrdinal,GlobalOrdinal,Node> HierarchyManager;
#else
typedef MueLu::HierarchyManager<Scalar,Node> HierarchyManager;
#endif
#endif
#ifdef MUELU_HIERARCHYFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::HierarchyFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> HierarchyFactory;
#else
typedef MueLu::HierarchyFactory<Scalar,Node> HierarchyFactory;
#endif
#endif
#ifdef MUELU_HIERARCHYUTILS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::HierarchyUtils<Scalar,LocalOrdinal,GlobalOrdinal,Node> HierarchyUtils;
#else
typedef MueLu::HierarchyUtils<Scalar,Node> HierarchyUtils;
#endif
#endif
#ifdef MUELU_IFPACK2SMOOTHER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::Ifpack2Smoother<Scalar,LocalOrdinal,GlobalOrdinal,Node> Ifpack2Smoother;
#else
typedef MueLu::Ifpack2Smoother<Scalar,Node> Ifpack2Smoother;
#endif
#endif
#ifdef MUELU_INDEFBLOCKEDDIAGONALSMOOTHER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::IndefBlockedDiagonalSmoother<Scalar,LocalOrdinal,GlobalOrdinal,Node> IndefBlockedDiagonalSmoother;
#else
typedef MueLu::IndefBlockedDiagonalSmoother<Scalar,Node> IndefBlockedDiagonalSmoother;
#endif
#endif
#ifdef MUELU_INTREPIDPCOARSENFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::IntrepidPCoarsenFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> IntrepidPCoarsenFactory;
#else
typedef MueLu::IntrepidPCoarsenFactory<Scalar,Node> IntrepidPCoarsenFactory;
#endif
#endif
#ifdef MUELU_LINEDETECTIONFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::LineDetectionFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> LineDetectionFactory;
#else
typedef MueLu::LineDetectionFactory<Scalar,Node> LineDetectionFactory;
#endif
#endif
#ifdef MUELU_LOCALPERMUTATIONSTRATEGY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::LocalPermutationStrategy<Scalar,LocalOrdinal,GlobalOrdinal,Node> LocalPermutationStrategy;
#else
typedef MueLu::LocalPermutationStrategy<Scalar,Node> LocalPermutationStrategy;
#endif
#endif
#ifdef MUELU_MAPTRANSFERFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::MapTransferFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> MapTransferFactory;
#else
typedef MueLu::MapTransferFactory<Scalar,Node> MapTransferFactory;
#endif
#endif
#ifdef MUELU_MATRIXANALYSISFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::MatrixAnalysisFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> MatrixAnalysisFactory;
#else
typedef MueLu::MatrixAnalysisFactory<Scalar,Node> MatrixAnalysisFactory;
#endif
#endif
#ifdef MUELU_MERGEDBLOCKEDMATRIXFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::MergedBlockedMatrixFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> MergedBlockedMatrixFactory;
#else
typedef MueLu::MergedBlockedMatrixFactory<Scalar,Node> MergedBlockedMatrixFactory;
#endif
#endif
#ifdef MUELU_MERGEDSMOOTHER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::MergedSmoother<Scalar,LocalOrdinal,GlobalOrdinal,Node> MergedSmoother;
#else
typedef MueLu::MergedSmoother<Scalar,Node> MergedSmoother;
#endif
#endif
#ifdef MUELU_MULTIVECTORTRANSFERFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::MultiVectorTransferFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> MultiVectorTransferFactory;
#else
typedef MueLu::MultiVectorTransferFactory<Scalar,Node> MultiVectorTransferFactory;
#endif
#endif
#ifdef MUELU_NULLSPACEFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::NullspaceFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> NullspaceFactory;
#else
typedef MueLu::NullspaceFactory<Scalar,Node> NullspaceFactory;
#endif
#endif
#ifdef MUELU_NULLSPACEFACTORY_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::NullspaceFactory_kokkos<Scalar,LocalOrdinal,GlobalOrdinal,Node> NullspaceFactory_kokkos;
#else
typedef MueLu::NullspaceFactory_kokkos<Scalar,Node> NullspaceFactory_kokkos;
#endif
#endif
#ifdef MUELU_NULLSPACEPRESMOOTHFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::NullspacePresmoothFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> NullspacePresmoothFactory;
#else
typedef MueLu::NullspacePresmoothFactory<Scalar,Node> NullspacePresmoothFactory;
#endif
#endif
#ifdef MUELU_PATTERNFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::PatternFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> PatternFactory;
#else
typedef MueLu::PatternFactory<Scalar,Node> PatternFactory;
#endif
#endif
#ifdef MUELU_PERFUTILS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::PerfUtils<Scalar,LocalOrdinal,GlobalOrdinal,Node> PerfUtils;
#else
typedef MueLu::PerfUtils<Scalar,Node> PerfUtils;
#endif
#endif
#ifdef MUELU_PERMUTATIONFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::PermutationFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> PermutationFactory;
#else
typedef MueLu::PermutationFactory<Scalar,Node> PermutationFactory;
#endif
#endif
#ifdef MUELU_PERMUTINGSMOOTHER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::PermutingSmoother<Scalar,LocalOrdinal,GlobalOrdinal,Node> PermutingSmoother;
#else
typedef MueLu::PermutingSmoother<Scalar,Node> PermutingSmoother;
#endif
#endif
#ifdef MUELU_PGPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::PgPFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> PgPFactory;
#else
typedef MueLu::PgPFactory<Scalar,Node> PgPFactory;
#endif
#endif
#ifdef MUELU_PREDROPFUNCTIONBASECLASS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::PreDropFunctionBaseClass<Scalar,LocalOrdinal,GlobalOrdinal,Node> PreDropFunctionBaseClass;
#else
typedef MueLu::PreDropFunctionBaseClass<Scalar,Node> PreDropFunctionBaseClass;
#endif
#endif
#ifdef MUELU_PREDROPFUNCTIONCONSTVAL_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::PreDropFunctionConstVal<Scalar,LocalOrdinal,GlobalOrdinal,Node> PreDropFunctionConstVal;
#else
typedef MueLu::PreDropFunctionConstVal<Scalar,Node> PreDropFunctionConstVal;
#endif
#endif
#ifdef MUELU_PROJECTORSMOOTHER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::ProjectorSmoother<Scalar,LocalOrdinal,GlobalOrdinal,Node> ProjectorSmoother;
#else
typedef MueLu::ProjectorSmoother<Scalar,Node> ProjectorSmoother;
#endif
#endif
#ifdef MUELU_RAPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::RAPFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> RAPFactory;
#else
typedef MueLu::RAPFactory<Scalar,Node> RAPFactory;
#endif
#endif
#ifdef MUELU_RAPSHIFTFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::RAPShiftFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> RAPShiftFactory;
#else
typedef MueLu::RAPShiftFactory<Scalar,Node> RAPShiftFactory;
#endif
#endif
#ifdef MUELU_REBALANCEACFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::RebalanceAcFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> RebalanceAcFactory;
#else
typedef MueLu::RebalanceAcFactory<Scalar,Node> RebalanceAcFactory;
#endif
#endif
#ifdef MUELU_REBALANCEBLOCKACFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::RebalanceBlockAcFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> RebalanceBlockAcFactory;
#else
typedef MueLu::RebalanceBlockAcFactory<Scalar,Node> RebalanceBlockAcFactory;
#endif
#endif
#ifdef MUELU_REBALANCEBLOCKINTERPOLATIONFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::RebalanceBlockInterpolationFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> RebalanceBlockInterpolationFactory;
#else
typedef MueLu::RebalanceBlockInterpolationFactory<Scalar,Node> RebalanceBlockInterpolationFactory;
#endif
#endif
#ifdef MUELU_REBALANCEBLOCKRESTRICTIONFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::RebalanceBlockRestrictionFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> RebalanceBlockRestrictionFactory;
#else
typedef MueLu::RebalanceBlockRestrictionFactory<Scalar,Node> RebalanceBlockRestrictionFactory;
#endif
#endif
#ifdef MUELU_REBALANCETRANSFERFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::RebalanceTransferFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> RebalanceTransferFactory;
#else
typedef MueLu::RebalanceTransferFactory<Scalar,Node> RebalanceTransferFactory;
#endif
#endif
#ifdef MUELU_REORDERBLOCKAFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::ReorderBlockAFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> ReorderBlockAFactory;
#else
typedef MueLu::ReorderBlockAFactory<Scalar,Node> ReorderBlockAFactory;
#endif
#endif
#ifdef MUELU_REPARTITIONFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::RepartitionFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> RepartitionFactory;
#else
typedef MueLu::RepartitionFactory<Scalar,Node> RepartitionFactory;
#endif
#endif
#ifdef MUELU_REPARTITIONBLOCKDIAGONALFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::RepartitionBlockDiagonalFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> RepartitionBlockDiagonalFactory;
#else
typedef MueLu::RepartitionBlockDiagonalFactory<Scalar,Node> RepartitionBlockDiagonalFactory;
#endif
#endif
#ifdef MUELU_REPARTITIONHEURISTICFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::RepartitionHeuristicFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> RepartitionHeuristicFactory;
#else
typedef MueLu::RepartitionHeuristicFactory<Scalar,Node> RepartitionHeuristicFactory;
#endif
#endif
#ifdef MUELU_RIGIDBODYMODEFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::RigidBodyModeFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> RigidBodyModeFactory;
#else
typedef MueLu::RigidBodyModeFactory<Scalar,Node> RigidBodyModeFactory;
#endif
#endif
#ifdef MUELU_SAPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::SaPFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> SaPFactory;
#else
typedef MueLu::SaPFactory<Scalar,Node> SaPFactory;
#endif
#endif
#ifdef MUELU_SAPFACTORY_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::SaPFactory_kokkos<Scalar,LocalOrdinal,GlobalOrdinal,Node> SaPFactory_kokkos;
#else
typedef MueLu::SaPFactory_kokkos<Scalar,Node> SaPFactory_kokkos;
#endif
#endif
#ifdef MUELU_SCALEDNULLSPACEFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::ScaledNullspaceFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> ScaledNullspaceFactory;
#else
typedef MueLu::ScaledNullspaceFactory<Scalar,Node> ScaledNullspaceFactory;
#endif
#endif
#ifdef MUELU_SCHURCOMPLEMENTFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::SchurComplementFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> SchurComplementFactory;
#else
typedef MueLu::SchurComplementFactory<Scalar,Node> SchurComplementFactory;
#endif
#endif
#ifdef MUELU_SEGREGATEDAFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::SegregatedAFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> SegregatedAFactory;
#else
typedef MueLu::SegregatedAFactory<Scalar,Node> SegregatedAFactory;
#endif
#endif
#ifdef MUELU_SHIFTEDLAPLACIAN_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::ShiftedLaplacian<Scalar,LocalOrdinal,GlobalOrdinal,Node> ShiftedLaplacian;
#else
typedef MueLu::ShiftedLaplacian<Scalar,Node> ShiftedLaplacian;
#endif
#endif
#ifdef MUELU_SHIFTEDLAPLACIANOPERATOR_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::ShiftedLaplacianOperator<Scalar,LocalOrdinal,GlobalOrdinal,Node> ShiftedLaplacianOperator;
#else
typedef MueLu::ShiftedLaplacianOperator<Scalar,Node> ShiftedLaplacianOperator;
#endif
#endif
#ifdef MUELU_SIMPLESMOOTHER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::SimpleSmoother<Scalar,LocalOrdinal,GlobalOrdinal,Node> SimpleSmoother;
#else
typedef MueLu::SimpleSmoother<Scalar,Node> SimpleSmoother;
#endif
#endif
#ifdef MUELU_SMOOTHER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::Smoother<Scalar,LocalOrdinal,GlobalOrdinal,Node> Smoother;
#else
typedef MueLu::Smoother<Scalar,Node> Smoother;
#endif
#endif
#ifdef MUELU_SMOOTHERBASE_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::SmootherBase<Scalar,LocalOrdinal,GlobalOrdinal,Node> SmootherBase;
#else
typedef MueLu::SmootherBase<Scalar,Node> SmootherBase;
#endif
#endif
#ifdef MUELU_SMOOTHERFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::SmootherFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> SmootherFactory;
#else
typedef MueLu::SmootherFactory<Scalar,Node> SmootherFactory;
#endif
#endif
#ifdef MUELU_SMOOTHERPROTOTYPE_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::SmootherPrototype<Scalar,LocalOrdinal,GlobalOrdinal,Node> SmootherPrototype;
#else
typedef MueLu::SmootherPrototype<Scalar,Node> SmootherPrototype;
#endif
#endif
#ifdef MUELU_SOLVERBASE_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::SolverBase<Scalar,LocalOrdinal,GlobalOrdinal,Node> SolverBase;
#else
typedef MueLu::SolverBase<Scalar,Node> SolverBase;
#endif
#endif
#ifdef MUELU_STEEPESTDESCENTSOLVER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::SteepestDescentSolver<Scalar,LocalOrdinal,GlobalOrdinal,Node> SteepestDescentSolver;
#else
typedef MueLu::SteepestDescentSolver<Scalar,Node> SteepestDescentSolver;
#endif
#endif
#ifdef MUELU_STRUCTUREDAGGREGATIONFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::StructuredAggregationFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> StructuredAggregationFactory;
#else
typedef MueLu::StructuredAggregationFactory<Scalar,Node> StructuredAggregationFactory;
#endif
#endif
#ifdef MUELU_STRUCTUREDLINEDETECTIONFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::StructuredLineDetectionFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> StructuredLineDetectionFactory;
#else
typedef MueLu::StructuredLineDetectionFactory<Scalar,Node> StructuredLineDetectionFactory;
#endif
#endif
#ifdef MUELU_SUBBLOCKAFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::SubBlockAFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> SubBlockAFactory;
#else
typedef MueLu::SubBlockAFactory<Scalar,Node> SubBlockAFactory;
#endif
#endif
#ifdef MUELU_TEKOSMOOTHER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::TekoSmoother<Scalar,LocalOrdinal,GlobalOrdinal,Node> TekoSmoother;
#else
typedef MueLu::TekoSmoother<Scalar,Node> TekoSmoother;
#endif
#endif
#ifdef MUELU_TENTATIVEPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::TentativePFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> TentativePFactory;
#else
typedef MueLu::TentativePFactory<Scalar,Node> TentativePFactory;
#endif
#endif
#ifdef MUELU_TENTATIVEPFACTORY_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::TentativePFactory_kokkos<Scalar,LocalOrdinal,GlobalOrdinal,Node> TentativePFactory_kokkos;
#else
typedef MueLu::TentativePFactory_kokkos<Scalar,Node> TentativePFactory_kokkos;
#endif
#endif
#ifdef MUELU_THRESHOLDAFILTERFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::ThresholdAFilterFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> ThresholdAFilterFactory;
#else
typedef MueLu::ThresholdAFilterFactory<Scalar,Node> ThresholdAFilterFactory;
#endif
#endif
#ifdef MUELU_TOGGLECOORDINATESTRANSFERFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::ToggleCoordinatesTransferFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> ToggleCoordinatesTransferFactory;
#else
typedef MueLu::ToggleCoordinatesTransferFactory<Scalar,Node> ToggleCoordinatesTransferFactory;
#endif
#endif
#ifdef MUELU_TOGGLEPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::TogglePFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> TogglePFactory;
#else
typedef MueLu::TogglePFactory<Scalar,Node> TogglePFactory;
#endif
#endif
#ifdef MUELU_TOPRAPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::TopRAPFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> TopRAPFactory;
#else
typedef MueLu::TopRAPFactory<Scalar,Node> TopRAPFactory;
#endif
#endif
#ifdef MUELU_TOPSMOOTHERFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::TopSmootherFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> TopSmootherFactory;
#else
typedef MueLu::TopSmootherFactory<Scalar,Node> TopSmootherFactory;
#endif
#endif
#ifdef MUELU_TPETRAOPERATOR_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::TpetraOperator<Scalar,LocalOrdinal,GlobalOrdinal,Node> TpetraOperator;
#else
typedef MueLu::TpetraOperator<Scalar,Node> TpetraOperator;
#endif
#endif
#ifdef MUELU_TRANSPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::TransPFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> TransPFactory;
#else
typedef MueLu::TransPFactory<Scalar,Node> TransPFactory;
#endif
#endif
#ifdef MUELU_TRILINOSSMOOTHER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::TrilinosSmoother<Scalar,LocalOrdinal,GlobalOrdinal,Node> TrilinosSmoother;
#else
typedef MueLu::TrilinosSmoother<Scalar,Node> TrilinosSmoother;
#endif
#endif
#ifdef MUELU_UNSMOOSHFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::UnsmooshFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> UnsmooshFactory;
#else
typedef MueLu::UnsmooshFactory<Scalar,Node> UnsmooshFactory;
#endif
#endif
#ifdef MUELU_USERPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::UserPFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> UserPFactory;
#else
typedef MueLu::UserPFactory<Scalar,Node> UserPFactory;
#endif
#endif
#ifdef MUELU_UTILITIES_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::Utilities<Scalar,LocalOrdinal,GlobalOrdinal,Node> Utilities;
#else
typedef MueLu::Utilities<Scalar,Node> Utilities;
#endif
#endif
#ifdef MUELU_UTILITIESBASE_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node> UtilitiesBase;
#else
typedef MueLu::UtilitiesBase<Scalar,Node> UtilitiesBase;
#endif
#endif
#ifdef MUELU_UTILITIES_KOKKOS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::Utilities_kokkos<Scalar,LocalOrdinal,GlobalOrdinal,Node> Utilities_kokkos;
#else
typedef MueLu::Utilities_kokkos<Scalar,Node> Utilities_kokkos;
#endif
#endif
#ifdef MUELU_VARIABLEDOFLAPLACIANFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::VariableDofLaplacianFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> VariableDofLaplacianFactory;
#else
typedef MueLu::VariableDofLaplacianFactory<Scalar,Node> VariableDofLaplacianFactory;
#endif
#endif
#ifdef MUELU_SEMICOARSENPFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::SemiCoarsenPFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> SemiCoarsenPFactory;
#else
typedef MueLu::SemiCoarsenPFactory<Scalar,Node> SemiCoarsenPFactory;
#endif
#endif
#ifdef MUELU_UZAWASMOOTHER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::UzawaSmoother<Scalar,LocalOrdinal,GlobalOrdinal,Node> UzawaSmoother;
#else
typedef MueLu::UzawaSmoother<Scalar,Node> UzawaSmoother;
#endif
#endif
#ifdef MUELU_VISUALIZATIONHELPERS_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::VisualizationHelpers<Scalar,LocalOrdinal,GlobalOrdinal,Node> VisualizationHelpers;
#else
typedef MueLu::VisualizationHelpers<Scalar,Node> VisualizationHelpers;
#endif
#endif
#ifdef MUELU_ZOLTANINTERFACE_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::ZoltanInterface<Scalar,LocalOrdinal,GlobalOrdinal,Node> ZoltanInterface;
#else
typedef MueLu::ZoltanInterface<Scalar,Node> ZoltanInterface;
#endif
#endif
#ifdef MUELU_ZOLTAN2INTERFACE_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::Zoltan2Interface<Scalar,LocalOrdinal,GlobalOrdinal,Node> Zoltan2Interface;
#else
typedef MueLu::Zoltan2Interface<Scalar,Node> Zoltan2Interface;
#endif
#endif
#ifdef MUELU_NODEPARTITIONINTERFACE_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::NodePartitionInterface<Scalar,LocalOrdinal,GlobalOrdinal,Node> NodePartitionInterface;
#else
typedef MueLu::NodePartitionInterface<Scalar,Node> NodePartitionInterface;
#endif
#endif
#ifdef MUELU_ADAPTIVESAMLPARAMETERLISTINTERPRETER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::AdaptiveSaMLParameterListInterpreter<Scalar,LocalOrdinal,GlobalOrdinal,Node> AdaptiveSaMLParameterListInterpreter;
#else
typedef MueLu::AdaptiveSaMLParameterListInterpreter<Scalar,Node> AdaptiveSaMLParameterListInterpreter;
#endif
#endif
#ifdef MUELU_FACTORYFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::FactoryFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> FactoryFactory;
#else
typedef MueLu::FactoryFactory<Scalar,Node> FactoryFactory;
#endif
#endif
#ifdef MUELU_MLPARAMETERLISTINTERPRETER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::MLParameterListInterpreter<Scalar,LocalOrdinal,GlobalOrdinal,Node> MLParameterListInterpreter;
#else
typedef MueLu::MLParameterListInterpreter<Scalar,Node> MLParameterListInterpreter;
#endif
#endif
#ifdef MUELU_PARAMETERLISTINTERPRETER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::ParameterListInterpreter<Scalar,LocalOrdinal,GlobalOrdinal,Node> ParameterListInterpreter;
#else
typedef MueLu::ParameterListInterpreter<Scalar,Node> ParameterListInterpreter;
#endif
#endif
#ifdef MUELU_TWOLEVELMATLABFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::TwoLevelMatlabFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> TwoLevelMatlabFactory;
#else
typedef MueLu::TwoLevelMatlabFactory<Scalar,Node> TwoLevelMatlabFactory;
#endif
#endif
#ifdef MUELU_SINGLELEVELMATLABFACTORY_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::SingleLevelMatlabFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node> SingleLevelMatlabFactory;
#else
typedef MueLu::SingleLevelMatlabFactory<Scalar,Node> SingleLevelMatlabFactory;
#endif
#endif
#ifdef MUELU_MATLABSMOOTHER_SHORT
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
typedef MueLu::MatlabSmoother<Scalar,LocalOrdinal,GlobalOrdinal,Node> MatlabSmoother;
#else
typedef MueLu::MatlabSmoother<Scalar,Node> MatlabSmoother;
#endif
#endif
