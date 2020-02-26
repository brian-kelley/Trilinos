//@HEADER
// ************************************************************************
//
//               ShyLU: Hybrid preconditioner package
//                 Copyright 2012 Sandia Corporation
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
// Questions? Contact Alexander Heinlein (alexander.heinlein@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#ifndef _FROSCH_SCHWARZOPERATOR_DECL_HPP
#define _FROSCH_SCHWARZOPERATOR_DECL_HPP

#include <Xpetra_MatrixMatrix.hpp>
#include <Xpetra_TripleMatrixMultiply.hpp>
#include <Xpetra_Export.hpp>

#include <Teuchos_DefaultSerialComm.hpp>

#include <ShyLU_DDFROSch_config.h>

#include <FROSch_DDInterface_def.hpp>
#include <FROSch_EntitySet_def.hpp>

#include <FROSch_CoarseSpace_def.hpp>
#include <FROSch_InterfacePartitionOfUnity_def.hpp>
#include <FROSch_LocalPartitionOfUnityBasis_def.hpp>

#include <FROSch_SubdomainSolver_def.hpp>

// TODO: Auf const überprüfen
// TODO: #ifndef überprüfen ??????


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    class Solver;

    template <class SC = double,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
              class LO = int,
              class GO = DefaultGlobalOrdinal,
#endif
              class NO = KokkosClassic::DefaultNode::DefaultNodeType>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    class SchwarzOperator : public Operator<SC,LO,GO,NO> {
#else
    class SchwarzOperator : public Operator<SC,NO> {
#endif

    protected:

#ifndef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using LO = typename Tpetra::Map<>::local_ordinal_type;
        using GO = typename Tpetra::Map<>::global_ordinal_type;
#endif
        using CommPtr                           = RCP<const Comm<int> >;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMap                              = Map<LO,GO,NO>;
#else
        using XMap                              = Map<NO>;
#endif
        using XMapPtr                           = RCP<XMap>;
        using ConstXMapPtr                      = RCP<const XMap>;
        using XMapPtrVecPtr                     = ArrayRCP<XMapPtr>;
        using ConstXMapPtrVecPtr                = ArrayRCP<ConstXMapPtr>;
        using XMapPtrVecPtr2D                   = ArrayRCP<XMapPtrVecPtr>;
        using ConstXMapPtrVecPtr2D              = ArrayRCP<ConstXMapPtrVecPtr>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMatrix                           = Matrix<SC,LO,GO,NO>;
#else
        using XMatrix                           = Matrix<SC,NO>;
#endif
        using XMatrixPtr                        = RCP<XMatrix>;
        using ConstXMatrixPtr                   = RCP<const XMatrix>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XCrsGraph                         = CrsGraph<LO,GO,NO>;
#else
        using XCrsGraph                         = CrsGraph<NO>;
#endif
        using GraphPtr                          = RCP<XCrsGraph>;
        using ConstXCrsGraphPtr                 = RCP<const XCrsGraph>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMultiVector                      = MultiVector<SC,LO,GO,NO>;
#else
        using XMultiVector                      = MultiVector<SC,NO>;
#endif
        using XMultiVectorPtr                   = RCP<XMultiVector>;
        using ConstXMultiVectorPtr              = RCP<const XMultiVector>;
        using XMultiVectorPtrVecPtr             = ArrayRCP<XMultiVectorPtr>;
        using ConstXMultiVectorPtrVecPtr        = ArrayRCP<ConstXMultiVectorPtr>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XImport                           = Import<LO,GO,NO>;
#else
        using XImport                           = Import<NO>;
#endif
        using XImportPtr                        = RCP<XImport>;
        using XImportPtrVecPtr                  = ArrayRCP<XImportPtr>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XExport                           = Export<LO,GO,NO>;
#else
        using XExport                           = Export<NO>;
#endif
        using XExportPtr                        = RCP<XExport>;
        using XExportPtrVecPtr                  = ArrayRCP<XExportPtr>;

        using ParameterListPtr                  = RCP<ParameterList>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using DDInterfacePtr                    = RCP<DDInterface<SC,LO,GO,NO> >;
#else
        using DDInterfacePtr                    = RCP<DDInterface<SC,NO> >;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using EntitySetPtr                      = RCP<EntitySet<SC,LO,GO,NO> >;
#else
        using EntitySetPtr                      = RCP<EntitySet<SC,NO> >;
#endif
        using EntitySetConstPtr                 = const EntitySetPtr;
        using EntitySetPtrVecPtr                = ArrayRCP<EntitySetPtr>;
        using EntitySetPtrConstVecPtr           = const EntitySetPtrVecPtr;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using CoarseSpacePtr                    = RCP<CoarseSpace<SC,LO,GO,NO> >;
#else
        using CoarseSpacePtr                    = RCP<CoarseSpace<SC,NO> >;
#endif
        using CoarseSpacePtrVecPtr              = ArrayRCP<CoarseSpacePtr>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using InterfaceEntityPtr                = RCP<InterfaceEntity<SC,LO,GO,NO> >;
#else
        using InterfaceEntityPtr                = RCP<InterfaceEntity<SC,NO> >;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using PartitionOfUnityPtr               = RCP<PartitionOfUnity<SC,LO,GO,NO> >;
        using InterfacePartitionOfUnityPtr      = RCP<InterfacePartitionOfUnity<SC,LO,GO,NO> >;
#else
        using PartitionOfUnityPtr               = RCP<PartitionOfUnity<SC,NO> >;
        using InterfacePartitionOfUnityPtr      = RCP<InterfacePartitionOfUnity<SC,NO> >;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using LocalPartitionOfUnityBasisPtr     = RCP<LocalPartitionOfUnityBasis<SC,LO,GO,NO> >;
#else
        using LocalPartitionOfUnityBasisPtr     = RCP<LocalPartitionOfUnityBasis<SC,NO> >;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using SchwarzOperatorPtr                = RCP<SchwarzOperator<SC,LO,GO,NO> >;
#else
        using SchwarzOperatorPtr                = RCP<SchwarzOperator<SC,NO> >;
#endif
        using SchwarzOperatorPtrVec             = Array<SchwarzOperatorPtr>;
        using SchwarzOperatorPtrVecPtr          = ArrayRCP<SchwarzOperatorPtr>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using SubdomainSolverPtr                = RCP<SubdomainSolver<SC,LO,GO,NO> >;
#else
        using SubdomainSolverPtr                = RCP<SubdomainSolver<SC,NO> >;
#endif

        using DofOrderingVecPtr                 = ArrayRCP<DofOrdering>;

        using UN                                = unsigned;
        using ConstUN                           = const UN;
        using UNVec                             = Array<UN>;
        using UNVecPtr                          = ArrayRCP<UN>;
        using ConstUNVecView                    = ArrayView<const UN>;

        using LOVec                             = Array<LO>;
        using LOVecPtr                          = ArrayRCP<LO>;
        using LOVecView                         = ArrayView<LO>;
        using ConstLOVecView                    = ArrayView<const LO>;
        using LOVecPtr2D                        = ArrayRCP<LOVecPtr>;

        using GOVec                             = Array<GO>;
        using GOVecPtr                          = ArrayRCP<GO>;
        using GOVecView                         = ArrayView<GO>;
        using ConstGOVecView                    = ArrayView<const GO>;
        using GOVec2D                           = Array<GOVec>;
        using GOVecPtr2D                        = ArrayRCP<GOVecPtr>;

        using SCVec                             = Array<SC>;
        using SCVecPtr                          = ArrayRCP<SC>;
        using ConstSCVecPtr                     = ArrayRCP<const SC>;
        using ConstSCVecView                    = ArrayView<const SC>;

        using BoolVec                           = Array<bool>;
        using BoolVecPtr                        = ArrayRCP<bool>;

    public:

        SchwarzOperator(CommPtr comm);

        SchwarzOperator(ConstXMatrixPtr k,
                        ParameterListPtr parameterList);

        virtual ~SchwarzOperator();

        virtual int initialize() = 0;

        virtual int compute() = 0;

        // Y = alpha * A^mode * X + beta * Y
        virtual void apply(const XMultiVector &x,
                           XMultiVector &y,
                           ETransp mode=NO_TRANS,
                           SC alpha=ScalarTraits<SC>::one(),
                           SC beta=ScalarTraits<SC>::zero()) const;

        virtual void apply(const XMultiVector &x,
                           XMultiVector &y,
                           bool usePreconditionerOnly,
                           ETransp mode=NO_TRANS,
                           SC alpha=ScalarTraits<SC>::one(),
                           SC beta=ScalarTraits<SC>::zero()) const = 0;

        virtual ConstXMapPtr getDomainMap() const;

        virtual ConstXMapPtr getRangeMap() const;

        virtual void describe(FancyOStream &out,
                              const EVerbosityLevel verbLevel=Describable::verbLevel_default) const = 0;

        virtual std::string description() const = 0;

        bool isInitialized() const;

        bool isComputed() const;

        int resetMatrix(ConstXMatrixPtr &k);

      
        virtual void residual(const XMultiVector & X,
                              const XMultiVector & B,
                              XMultiVector& R) const;

    protected:

        CommPtr MpiComm_;
        CommPtr SerialComm_ = createSerialComm<int>();

        ConstXMatrixPtr K_;

        ParameterListPtr ParameterList_;

        bool Verbose_ = false;

        bool IsInitialized_ = false;
        bool IsComputed_ = false;

        ConstUN LevelID_ = 1;
    };

}

#endif
