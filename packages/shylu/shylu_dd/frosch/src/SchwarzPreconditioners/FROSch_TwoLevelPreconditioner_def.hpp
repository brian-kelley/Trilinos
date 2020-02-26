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

#ifndef _FROSCH_TWOLEVELPRECONDITIONER_DEF_HPP
#define _FROSCH_TWOLEVELPRECONDITIONER_DEF_HPP

#include <FROSch_TwoLevelPreconditioner_decl.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    TwoLevelPreconditioner<SC,LO,GO,NO>::TwoLevelPreconditioner(ConstXMatrixPtr k,
#else
    template <class SC,class NO>
    TwoLevelPreconditioner<SC,NO>::TwoLevelPreconditioner(ConstXMatrixPtr k,
#endif
                                                                ParameterListPtr parameterList) :
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    OneLevelPreconditioner<SC,LO,GO,NO> (k,parameterList)
#else
    OneLevelPreconditioner<SC,NO> (k,parameterList)
#endif
    {
        FROSCH_TIMER_START_LEVELID(twoLevelPreconditionerTime,"TwoLevelPreconditioner::TwoLevelPreconditioner::");
        if (!this->ParameterList_->get("CoarseOperator Type","IPOUHarmonicCoarseOperator").compare("IPOUHarmonicCoarseOperator")) {
            // Set the LevelID in the sublist
            parameterList->sublist("IPOUHarmonicCoarseOperator").set("Level ID",this->LevelID_);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            CoarseOperator_ = IPOUHarmonicCoarseOperatorPtr(new IPOUHarmonicCoarseOperator<SC,LO,GO,NO>(k,sublist(parameterList,"IPOUHarmonicCoarseOperator")));
#else
            CoarseOperator_ = IPOUHarmonicCoarseOperatorPtr(new IPOUHarmonicCoarseOperator<SC,NO>(k,sublist(parameterList,"IPOUHarmonicCoarseOperator")));
#endif
        } else if (!this->ParameterList_->get("CoarseOperator Type","IPOUHarmonicCoarseOperator").compare("GDSWCoarseOperator")) {
            // Set the LevelID in the sublist
            parameterList->sublist("GDSWCoarseOperator").set("Level ID",this->LevelID_);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            CoarseOperator_ = GDSWCoarseOperatorPtr(new GDSWCoarseOperator<SC,LO,GO,NO>(k,sublist(parameterList,"GDSWCoarseOperator")));
#else
            CoarseOperator_ = GDSWCoarseOperatorPtr(new GDSWCoarseOperator<SC,NO>(k,sublist(parameterList,"GDSWCoarseOperator")));
#endif
        } else if (!this->ParameterList_->get("CoarseOperator Type","IPOUHarmonicCoarseOperator").compare("RGDSWCoarseOperator")) {
            // Set the LevelID in the sublist
            parameterList->sublist("RGDSWCoarseOperator").set("Level ID",this->LevelID_);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            CoarseOperator_ = RGDSWCoarseOperatorPtr(new RGDSWCoarseOperator<SC,LO,GO,NO>(k,sublist(parameterList,"RGDSWCoarseOperator")));
#else
            CoarseOperator_ = RGDSWCoarseOperatorPtr(new RGDSWCoarseOperator<SC,NO>(k,sublist(parameterList,"RGDSWCoarseOperator")));
#endif
        } else {
            FROSCH_ASSERT(false,"CoarseOperator Type unkown.");
        } // TODO: Add ability to disable individual levels
        if (this->UseMultiplicative_) {
            this->MultiplicativeOperator_->addOperator(CoarseOperator_);
        }
        else{
            this->SumOperator_->addOperator(CoarseOperator_);
        }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int TwoLevelPreconditioner<SC,LO,GO,NO>::initialize(bool useDefaultParameters)
#else
    template <class SC,class NO>
    int TwoLevelPreconditioner<SC,NO>::initialize(bool useDefaultParameters)
#endif
    {
        if (useDefaultParameters) {
            return initialize(3,1,1);
        } else {
            return initialize(this->ParameterList_->get("Dimension",1),this->ParameterList_->get("DofsPerNode",1),this->ParameterList_->get("Overlap",1));
        }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int TwoLevelPreconditioner<SC,LO,GO,NO>::initialize(UN dimension,
#else
    template <class SC,class NO>
    int TwoLevelPreconditioner<SC,NO>::initialize(UN dimension,
#endif
                                                        int overlap,
                                                        UN dofsPerNode,
                                                        DofOrdering dofOrdering)
    {
        return initialize(dimension,dofsPerNode,overlap,null,null,dofOrdering,null,null,null);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int TwoLevelPreconditioner<SC,LO,GO,NO>::initialize(UN dimension,
#else
    template <class SC,class NO>
    int TwoLevelPreconditioner<SC,NO>::initialize(UN dimension,
#endif
                                                        int overlap,
                                                        ConstXMapPtr repeatedMap,
                                                        UN dofsPerNode,
                                                        DofOrdering dofOrdering,
                                                        ConstXMultiVectorPtr nodeList)
    {
        return initialize(dimension,dofsPerNode,overlap,null,nodeList,dofOrdering,repeatedMap,null,null);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int TwoLevelPreconditioner<SC,LO,GO,NO>::initialize(UN dimension,
#else
    template <class SC,class NO>
    int TwoLevelPreconditioner<SC,NO>::initialize(UN dimension,
#endif
                                                        UN dofsPerNode,
                                                        int overlap,
                                                        ConstXMultiVectorPtr nullSpaceBasis,
                                                        ConstXMultiVectorPtr nodeList,
                                                        DofOrdering dofOrdering,
                                                        ConstXMapPtr repeatedMap,
                                                        ConstXMapPtrVecPtr dofsMaps,
                                                        GOVecPtr dirichletBoundaryDofs)
    {
        FROSCH_TIMER_START_LEVELID(initializeTime,"TwoLevelPreconditioner::initialize");
        ////////////
        // Checks //
        ////////////
        FROSCH_ASSERT(dofOrdering == NodeWise || dofOrdering == DimensionWise || dofOrdering == Custom,"ERROR: Specify a valid DofOrdering.");
        int ret = 0;

        //////////
        // Maps //
        //////////
        if (repeatedMap.is_null()) {
            FROSCH_TIMER_START_LEVELID(buildRepeatedMapTime,"BuildRepeatedMap");
            repeatedMap = BuildRepeatedMap(this->K_->getCrsGraph()); // Todo: Achtung, die UniqueMap könnte unsinnig verteilt sein. Falls es eine repeatedMap gibt, sollte dann die uniqueMap neu gebaut werden können. In diesem Fall, sollte man das aber basierend auf der repeatedNodesMap tun
        }
        // Build dofsMaps and repeatedNodesMap
        ConstXMapPtr repeatedNodesMap;
        if (dofsMaps.is_null()) {
            FROSCH_TIMER_START_LEVELID(buildDofMapsTime,"BuildDofMaps");
            if (0>BuildDofMaps(repeatedMap,dofsPerNode,dofOrdering,repeatedNodesMap,dofsMaps)) ret -= 100; // Todo: Rückgabewerte
        } else {
            FROSCH_ASSERT(dofsMaps.size()==dofsPerNode,"dofsMaps.size()!=dofsPerNode");
            for (UN i=0; i<dofsMaps.size(); i++) {
                FROSCH_ASSERT(!dofsMaps[i].is_null(),"dofsMaps[i].is_null()");
            }
            if (repeatedNodesMap.is_null()) {
                repeatedNodesMap = dofsMaps[0];
            }
        }

        //////////////////////////
        // Communicate nodeList //
        //////////////////////////
        if (!nodeList.is_null()) {
            FROSCH_TIMER_START_LEVELID(communicateNodeListTime,"Communicate Node List");
            ConstXMapPtr nodeListMap = nodeList->getMap();
            if (!nodeListMap->isSameAs(*repeatedNodesMap)) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                RCP<MultiVector<SC,LO,GO,NO> > tmpNodeList = MultiVectorFactory<SC,LO,GO,NO>::Build(repeatedNodesMap,nodeList->getNumVectors());
                RCP<Import<LO,GO,NO> > scatter = ImportFactory<LO,GO,NO>::Build(nodeListMap,repeatedNodesMap);
#else
                RCP<MultiVector<SC,NO> > tmpNodeList = MultiVectorFactory<SC,NO>::Build(repeatedNodesMap,nodeList->getNumVectors());
                RCP<Import<NO> > scatter = ImportFactory<NO>::Build(nodeListMap,repeatedNodesMap);
#endif
                tmpNodeList->doImport(*nodeList,*scatter,INSERT);
                nodeList = tmpNodeList.getConst();
            }
        }

        /////////////////////////////////////
        // Determine dirichletBoundaryDofs //
        /////////////////////////////////////
        if (dirichletBoundaryDofs.is_null()) {
            FROSCH_TIMER_START_LEVELID(determineDirichletRowsTime,"Determine Dirichlet Rows");
#ifdef FindOneEntryOnlyRowsGlobal_Matrix
            GOVecPtr dirichletBoundaryDofs = FindOneEntryOnlyRowsGlobal(this->K_.getConst(),repeatedMap);
#else
            GOVecPtr dirichletBoundaryDofs = FindOneEntryOnlyRowsGlobal(this->K_->getCrsGraph(),repeatedMap);
#endif
        }

        ////////////////////////////////////
        // Initialize OverlappingOperator //
        ////////////////////////////////////
        if (!this->ParameterList_->get("OverlappingOperator Type","AlgebraicOverlappingOperator").compare("AlgebraicOverlappingOperator")) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            AlgebraicOverlappingOperatorPtr algebraicOverlappigOperator = rcp_static_cast<AlgebraicOverlappingOperator<SC,LO,GO,NO> >(this->OverlappingOperator_);
#else
            AlgebraicOverlappingOperatorPtr algebraicOverlappigOperator = rcp_static_cast<AlgebraicOverlappingOperator<SC,NO> >(this->OverlappingOperator_);
#endif
            if (0>algebraicOverlappigOperator->initialize(overlap,repeatedMap)) ret -= 1;
        } else {
            FROSCH_ASSERT(false,"OverlappingOperator Type unkown.");
        }

        ///////////////////////////////
        // Initialize CoarseOperator //
        ///////////////////////////////
        if (!this->ParameterList_->get("CoarseOperator Type","IPOUHarmonicCoarseOperator").compare("IPOUHarmonicCoarseOperator")) {
            // Build Null Space
            if (!this->ParameterList_->get("Null Space Type","Laplace").compare("Laplace")) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                nullSpaceBasis = BuildNullSpace<SC,LO,GO,NO>(dimension,LaplaceNullSpace,repeatedMap,dofsPerNode,dofsMaps);
#else
                nullSpaceBasis = BuildNullSpace<SC,NO>(dimension,LaplaceNullSpace,repeatedMap,dofsPerNode,dofsMaps);
#endif
            } else if (!this->ParameterList_->get("Null Space Type","Laplace").compare("Linear Elasticity")) {
                nullSpaceBasis = BuildNullSpace(dimension,LinearElasticityNullSpace,repeatedMap,dofsPerNode,dofsMaps,nodeList);
            } else if (!this->ParameterList_->get("Null Space Type","Laplace").compare("Input")) {
                FROSCH_ASSERT(!nullSpaceBasis.is_null(),"Null Space Type is 'Input', but nullSpaceBasis.is_null().");
                ConstXMapPtr nullSpaceBasisMap = nullSpaceBasis->getMap();
                if (!nullSpaceBasisMap->isSameAs(*repeatedMap)) {
                    FROSCH_TIMER_START_LEVELID(communicateNullSpaceBasis,"Communicate Null Space");
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                    RCP<MultiVector<SC,LO,GO,NO> > tmpNullSpaceBasis = MultiVectorFactory<SC,LO,GO,NO>::Build(repeatedMap,nullSpaceBasis->getNumVectors());
                    RCP<Import<LO,GO,NO> > scatter = ImportFactory<LO,GO,NO>::Build(nullSpaceBasisMap,repeatedMap);
#else
                    RCP<MultiVector<SC,NO> > tmpNullSpaceBasis = MultiVectorFactory<SC,NO>::Build(repeatedMap,nullSpaceBasis->getNumVectors());
                    RCP<Import<NO> > scatter = ImportFactory<NO>::Build(nullSpaceBasisMap,repeatedMap);
#endif
                    tmpNullSpaceBasis->doImport(*nullSpaceBasis,*scatter,INSERT);
                    nullSpaceBasis = tmpNullSpaceBasis.getConst();
                }
            } else {
                FROSCH_ASSERT(false,"Null Space Type unknown.");
            }
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            IPOUHarmonicCoarseOperatorPtr iPOUHarmonicCoarseOperator = rcp_static_cast<IPOUHarmonicCoarseOperator<SC,LO,GO,NO> >(CoarseOperator_);
#else
            IPOUHarmonicCoarseOperatorPtr iPOUHarmonicCoarseOperator = rcp_static_cast<IPOUHarmonicCoarseOperator<SC,NO> >(CoarseOperator_);
#endif
            if (0>iPOUHarmonicCoarseOperator->initialize(dimension,dofsPerNode,repeatedNodesMap,dofsMaps,nullSpaceBasis,nodeList,dirichletBoundaryDofs)) ret -=10;
        } else if (!this->ParameterList_->get("CoarseOperator Type","IPOUHarmonicCoarseOperator").compare("GDSWCoarseOperator")) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            GDSWCoarseOperatorPtr gDSWCoarseOperator = rcp_static_cast<GDSWCoarseOperator<SC,LO,GO,NO> >(CoarseOperator_);
#else
            GDSWCoarseOperatorPtr gDSWCoarseOperator = rcp_static_cast<GDSWCoarseOperator<SC,NO> >(CoarseOperator_);
#endif
            if (0>gDSWCoarseOperator->initialize(dimension,dofsPerNode,repeatedNodesMap,dofsMaps,dirichletBoundaryDofs,nodeList)) ret -=10;
        } else if (!this->ParameterList_->get("CoarseOperator Type","IPOUHarmonicCoarseOperator").compare("RGDSWCoarseOperator")) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            RGDSWCoarseOperatorPtr rGDSWCoarseOperator = rcp_static_cast<RGDSWCoarseOperator<SC,LO,GO,NO> >(CoarseOperator_);
#else
            RGDSWCoarseOperatorPtr rGDSWCoarseOperator = rcp_static_cast<RGDSWCoarseOperator<SC,NO> >(CoarseOperator_);
#endif
            if (0>rGDSWCoarseOperator->initialize(dimension,dofsPerNode,repeatedNodesMap,dofsMaps,dirichletBoundaryDofs,nodeList)) ret -=10;
        } else {
            FROSCH_ASSERT(false,"CoarseOperator Type unkown.");
        }
        return ret;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int TwoLevelPreconditioner<SC,LO,GO,NO>::compute()
#else
    template <class SC,class NO>
    int TwoLevelPreconditioner<SC,NO>::compute()
#endif
    {
        FROSCH_TIMER_START_LEVELID(computeTime,"TwoLevelPreconditioner::compute");
        int ret = 0;
        if (0>this->OverlappingOperator_->compute()) ret -= 1;
        if (0>CoarseOperator_->compute()) ret -= 10;
        return ret;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    void TwoLevelPreconditioner<SC,LO,GO,NO>::describe(FancyOStream &out,
#else
    template <class SC,class NO>
    void TwoLevelPreconditioner<SC,NO>::describe(FancyOStream &out,
#endif
                                                       const EVerbosityLevel verbLevel) const
    {
        FROSCH_ASSERT(false,"describe() has to be implemented properly...");
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    std::string TwoLevelPreconditioner<SC,LO,GO,NO>::description() const
#else
    template <class SC,class NO>
    std::string TwoLevelPreconditioner<SC,NO>::description() const
#endif
    {
        return "GDSW Preconditioner";
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int TwoLevelPreconditioner<SC,LO,GO,NO>::resetMatrix(ConstXMatrixPtr &k)
#else
    template <class SC,class NO>
    int TwoLevelPreconditioner<SC,NO>::resetMatrix(ConstXMatrixPtr &k)
#endif
    {
        FROSCH_TIMER_START_LEVELID(resetMatrixTime,"TwoLevelPreconditioner::resetMatrix");
        this->K_ = k;
        this->OverlappingOperator_->resetMatrix(this->K_);
        CoarseOperator_->resetMatrix(this->K_);
        if (this->UseMultiplicative_) this->MultiplicativeOperator_->resetMatrix(this->K_);
        return 0;
    }
}

#endif
