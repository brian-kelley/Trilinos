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

#ifndef _FROSCH_COARSEOPERATOR_DECL_HPP
#define _FROSCH_COARSEOPERATOR_DECL_HPP

#include <FROSch_SchwarzOperator_def.hpp>

// #define FROSCH_COARSEOPERATOR_DETAIL_TIMERS
// #define FROSCH_COARSEOPERATOR_EXPORT_AND_IMPORT

// TODO: Member sortieren!?


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC = double,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
              class LO = int,
              class GO = DefaultGlobalOrdinal,
#endif
              class NO = KokkosClassic::DefaultNode::DefaultNodeType>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    class CoarseOperator : public SchwarzOperator<SC,LO,GO,NO> {
#else
    class CoarseOperator : public SchwarzOperator<SC,NO> {
#endif

    protected:

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using CommPtr               = typename SchwarzOperator<SC,LO,GO,NO>::CommPtr;
#else
        using LO = typename Tpetra::Map<>::local_ordinal_type;
        using GO = typename Tpetra::Map<>::global_ordinal_type;
        using CommPtr               = typename SchwarzOperator<SC,NO>::CommPtr;

        using XMap                  = typename SchwarzOperator<SC,NO>::XMap;
        using XMapPtr               = typename SchwarzOperator<SC,NO>::XMapPtr;
        using ConstXMapPtr          = typename SchwarzOperator<SC,NO>::ConstXMapPtr;
        using XMapPtrVecPtr         = typename SchwarzOperator<SC,NO>::XMapPtrVecPtr;
        using ConstXMapPtrVecPtr    = typename SchwarzOperator<SC,NO>::ConstXMapPtrVecPtr;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMap                  = typename SchwarzOperator<SC,LO,GO,NO>::XMap;
        using XMapPtr               = typename SchwarzOperator<SC,LO,GO,NO>::XMapPtr;
        using ConstXMapPtr          = typename SchwarzOperator<SC,LO,GO,NO>::ConstXMapPtr;
        using XMapPtrVecPtr         = typename SchwarzOperator<SC,LO,GO,NO>::XMapPtrVecPtr;
        using ConstXMapPtrVecPtr    = typename SchwarzOperator<SC,LO,GO,NO>::ConstXMapPtrVecPtr;
#else
        using XMatrixPtr            = typename SchwarzOperator<SC,NO>::XMatrixPtr;
        using ConstXMatrixPtr       = typename SchwarzOperator<SC,NO>::ConstXMatrixPtr;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMatrixPtr            = typename SchwarzOperator<SC,LO,GO,NO>::XMatrixPtr;
        using ConstXMatrixPtr       = typename SchwarzOperator<SC,LO,GO,NO>::ConstXMatrixPtr;
#else
        using XMultiVector          = typename SchwarzOperator<SC,NO>::XMultiVector;
        using XMultiVectorPtr       = typename SchwarzOperator<SC,NO>::XMultiVectorPtr;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMultiVector          = typename SchwarzOperator<SC,LO,GO,NO>::XMultiVector;
        using XMultiVectorPtr       = typename SchwarzOperator<SC,LO,GO,NO>::XMultiVectorPtr;

        using XImportPtrVecPtr      = typename SchwarzOperator<SC,LO,GO,NO>::XImportPtrVecPtr;
#else
        using XImportPtrVecPtr      = typename SchwarzOperator<SC,NO>::XImportPtrVecPtr;
#endif
        
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XExportPtrVecPtr      = typename SchwarzOperator<SC,LO,GO,NO>::XExportPtrVecPtr;
#else
        using XExportPtrVecPtr      = typename SchwarzOperator<SC,NO>::XExportPtrVecPtr;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using ParameterListPtr      = typename SchwarzOperator<SC,LO,GO,NO>::ParameterListPtr;
#else
        using ParameterListPtr      = typename SchwarzOperator<SC,NO>::ParameterListPtr;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using CoarseSpacePtr        = typename SchwarzOperator<SC,LO,GO,NO>::CoarseSpacePtr;
#else
        using CoarseSpacePtr        = typename SchwarzOperator<SC,NO>::CoarseSpacePtr;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using SubdomainSolverPtr    = typename SchwarzOperator<SC,LO,GO,NO>::SubdomainSolverPtr;
#else
        using SubdomainSolverPtr    = typename SchwarzOperator<SC,NO>::SubdomainSolverPtr;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using UN                    = typename SchwarzOperator<SC,LO,GO,NO>::UN;
#else
        using UN                    = typename SchwarzOperator<SC,NO>::UN;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using GOVec                 = typename SchwarzOperator<SC,LO,GO,NO>::GOVec;
        using GOVecPtr              = typename SchwarzOperator<SC,LO,GO,NO>::GOVecPtr;
#else
        using GOVec                 = typename SchwarzOperator<SC,NO>::GOVec;
        using GOVecPtr              = typename SchwarzOperator<SC,NO>::GOVecPtr;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using LOVec                 = typename SchwarzOperator<SC,LO,GO,NO>::LOVec;
        using LOVecPtr2D            = typename SchwarzOperator<SC,LO,GO,NO>::LOVecPtr2D;
#else
        using LOVec                 = typename SchwarzOperator<SC,NO>::LOVec;
        using LOVecPtr2D            = typename SchwarzOperator<SC,NO>::LOVecPtr2D;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using SCVec                 = typename SchwarzOperator<SC,LO,GO,NO>::SCVec;
#else
        using SCVec                 = typename SchwarzOperator<SC,NO>::SCVec;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using ConstLOVecView        = typename SchwarzOperator<SC,LO,GO,NO>::ConstLOVecView;
#else
        using ConstLOVecView        = typename SchwarzOperator<SC,NO>::ConstLOVecView;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using ConstGOVecView        = typename SchwarzOperator<SC,LO,GO,NO>::ConstGOVecView;
#else
        using ConstGOVecView        = typename SchwarzOperator<SC,NO>::ConstGOVecView;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using ConstSCVecView        = typename SchwarzOperator<SC,LO,GO,NO>::ConstSCVecView;
#else
        using ConstSCVecView        = typename SchwarzOperator<SC,NO>::ConstSCVecView;
#endif

    public:

        CoarseOperator(ConstXMatrixPtr k,
                       ParameterListPtr parameterList);

        ~CoarseOperator();

        virtual int initialize() = 0;

        virtual int compute();

        virtual XMapPtr computeCoarseSpace(CoarseSpacePtr coarseSpace) = 0;

        virtual int clearCoarseSpace();

        virtual void apply(const XMultiVector &x,
                           XMultiVector &y,
                           bool usePreconditionerOnly,
                           ETransp mode=NO_TRANS,
                           SC alpha=ScalarTraits<SC>::one(),
                           SC beta=ScalarTraits<SC>::zero()) const;

        virtual void applyPhiT(const XMultiVector& x,
                               XMultiVector& y) const;

        virtual void applyCoarseSolve(XMultiVector& x,
                                      XMultiVector& y,
                                      ETransp mode=NO_TRANS) const;

        virtual void applyPhi(const XMultiVector& x,
                              XMultiVector& y) const;

        virtual CoarseSpacePtr getCoarseSpace() const;

    protected:

        virtual int setUpCoarseOperator();

        XMatrixPtr buildCoarseMatrix();

        int buildCoarseSolveMap(ConstXMapPtr coarseMapUnique);


        CommPtr CoarseSolveComm_;

        bool OnCoarseSolveComm_ = false;
        
        int NumProcsCoarseSolve_ = 0;

        CoarseSpacePtr CoarseSpace_;

        XMatrixPtr Phi_;
        XMatrixPtr CoarseMatrix_;

        // Temp Vectors for apply()
        mutable XMultiVectorPtr XTmp_;
        mutable XMultiVectorPtr XCoarse_;
        mutable XMultiVectorPtr XCoarseSolve_;
        mutable XMultiVectorPtr XCoarseSolveTmp_;
        mutable XMultiVectorPtr YTmp_;
        mutable XMultiVectorPtr YCoarse_;
        mutable XMultiVectorPtr YCoarseSolve_;
        mutable XMultiVectorPtr YCoarseSolveTmp_;

        ConstXMapPtrVecPtr GatheringMaps_ = ConstXMapPtrVecPtr(0);
        XMapPtr CoarseSolveMap_;

        SubdomainSolverPtr CoarseSolver_;

        ParameterListPtr DistributionList_;

        XExportPtrVecPtr CoarseSolveExporters_ = XExportPtrVecPtr(0);
#ifdef FROSCH_COARSEOPERATOR_EXPORT_AND_IMPORT
        XImportPtrVecPtr CoarseSolveImporters_ = XImportPtrVecPtr(0);
#endif
    };

}

#endif
