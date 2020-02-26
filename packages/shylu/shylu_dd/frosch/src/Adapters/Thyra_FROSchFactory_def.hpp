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

#ifndef THYRA_FROSCH_XPETRA_FACTORY_DEF_HPP
#define THYRA_FROSCH_XPETRA_FACTORY_DEF_HPP

#include "Thyra_FROSchFactory_decl.hpp"


namespace Thyra {

    using namespace FROSch;
    using namespace Teuchos;
    using namespace Thyra;
    using namespace Xpetra;

    //Constructor
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    FROSchFactory<SC,LO,GO,NO>::FROSchFactory()
#else
    template <class SC, class NO>
    FROSchFactory<SC,NO>::FROSchFactory()
#endif
    {
        
    }

    //-----------------------------------------------------------
    //Check Type -> so far redundant
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    bool FROSchFactory<SC,LO,GO,NO>::isCompatible(const LinearOpSourceBase<SC>& fwdOpSrc) const
#else
    template <class SC, class NO>
    bool FROSchFactory<SC,NO>::isCompatible(const LinearOpSourceBase<SC>& fwdOpSrc) const
#endif
    {
        const ConstLinearOpBasePtr fwdOp = fwdOpSrc.getOp();
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        if (ThyraUtils<SC,LO,GO,NO>::isEpetra(fwdOp)) {
#else
        if (ThyraUtils<SC,NO>::isEpetra(fwdOp)) {
#endif
            return true;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        } else if (ThyraUtils<SC,LO,GO,NO>::isTpetra(fwdOp)) {
#else
        } else if (ThyraUtils<SC,NO>::isTpetra(fwdOp)) {
#endif
            return true;
        } else {
            return false;
        }
    }

    //--------------------------------------------------------------
    //Create Default Prec -> Not used here (maybe somewhere else?)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC, class LO, class GO , class NO>
    typename FROSchFactory<SC,LO,GO,NO>::PreconditionerBasePtr FROSchFactory<SC,LO,GO,NO>::createPrec() const
#else
    template<class SC, class NO>
    typename FROSchFactory<SC,NO>::PreconditionerBasePtr FROSchFactory<SC,NO>::createPrec() const
#endif
    {
        return rcp(new DefaultPreconditioner<SC>);
    }

    //-------------------------------------------------------------
    //Main Function to use FROSch as Prec
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC, class LO , class GO, class NO>
    void FROSchFactory<SC,LO,GO,NO>::initializePrec(const ConstLinearOpSourceBasePtr& fwdOpSrc,
#else
    template<class SC, class NO>
    void FROSchFactory<SC,NO>::initializePrec(const ConstLinearOpSourceBasePtr& fwdOpSrc,
#endif
                                                    PreconditionerBase<SC>* prec,
                                                    const ESupportSolveUse supportSolveUse) const
    {
        //PreCheck
        TEUCHOS_ASSERT(nonnull(fwdOpSrc));
        //TEUCHOS_ASSERT(this->isCompatible(*fwdOpSrc));
        TEUCHOS_ASSERT(prec);

        // Retrieve wrapped concrete Xpetra matrix from FwdOp
        const ConstLinearOpBasePtr fwdOp = fwdOpSrc->getOp();
        TEUCHOS_TEST_FOR_EXCEPT(is_null(fwdOp));

        // Check whether it is Epetra/Tpetra
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        bool bIsEpetra  = ThyraUtils<SC,LO,GO,NO>::isEpetra(fwdOp);
        bool bIsTpetra  = ThyraUtils<SC,LO,GO,NO>::isTpetra(fwdOp);
        bool bIsBlocked = ThyraUtils<SC,LO,GO,NO>::isBlockedOperator(fwdOp);
#else
        bool bIsEpetra  = ThyraUtils<SC,NO>::isEpetra(fwdOp);
        bool bIsTpetra  = ThyraUtils<SC,NO>::isTpetra(fwdOp);
        bool bIsBlocked = ThyraUtils<SC,NO>::isBlockedOperator(fwdOp);
#endif
        TEUCHOS_TEST_FOR_EXCEPT((bIsEpetra == true  && bIsTpetra == true));
        TEUCHOS_TEST_FOR_EXCEPT((bIsEpetra == bIsTpetra) && bIsBlocked == false);
        TEUCHOS_TEST_FOR_EXCEPT((bIsEpetra != bIsTpetra) && bIsBlocked == true);

        // Retrieve Matrix
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        ConstXCrsMatrixPtr xpetraFwdCrsMat = ThyraUtils<SC,LO,GO,NO>::toXpetra(fwdOp);
#else
        ConstXCrsMatrixPtr xpetraFwdCrsMat = ThyraUtils<SC,NO>::toXpetra(fwdOp);
#endif
        TEUCHOS_TEST_FOR_EXCEPT(is_null(xpetraFwdCrsMat));

        // AH 08/07/2019: Going from const to non-const to const. One should be able to improve this.
        XCrsMatrixPtr xpetraFwdCrsMatNonConst = rcp_const_cast<XCrsMatrix>(xpetraFwdCrsMat);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        XMatrixPtr ANonConst = rcp(new CrsMatrixWrap<SC,LO,GO,NO>(xpetraFwdCrsMatNonConst));
#else
        XMatrixPtr ANonConst = rcp(new CrsMatrixWrap<SC,NO>(xpetraFwdCrsMatNonConst));
#endif
        ConstXMatrixPtr A = ANonConst.getConst();
        
        CommPtr comm = A->getMap()->getComm();
        UnderlyingLib underlyingLib = A->getMap()->lib();

        // Retrieve concrete preconditioner object
        const Ptr<DefaultPreconditioner<SC> > defaultPrec = ptr(dynamic_cast<DefaultPreconditioner<SC> *>(prec));
        TEUCHOS_TEST_FOR_EXCEPT(is_null(defaultPrec));

        // extract preconditioner operator
        LinearOpBasePtr thyra_precOp = null;
        thyra_precOp = rcp_dynamic_cast<LinearOpBase<SC> >(defaultPrec->getNonconstUnspecifiedPrecOp(), true);

        // Abstract SchwarzPreconditioner
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        RCP<SchwarzPreconditioner<SC,LO,GO,NO> > SchwarzPreconditioner = null;
#else
        RCP<SchwarzPreconditioner<SC,NO> > SchwarzPreconditioner = null;
#endif
        
        const bool startingOver = (thyra_precOp.is_null() || !paramList_->isParameter("Recycling") || !paramList_->get("Recycling",true));
        
        if (startingOver) {
            FROSCH_ASSERT(paramList_->isParameter("FROSch Preconditioner Type"),"FROSch Preconditioner Type is not defined!");
            
            if (!paramList_->get("FROSch Preconditioner Type","TwoLevelPreconditioner").compare("AlgebraicOverlappingPreconditioner")) {
                // Extract the repeated map
                ConstXMapPtr repeatedMap = extractRepeatedMap(comm,underlyingLib);
                
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                RCP<AlgebraicOverlappingPreconditioner<SC,LO,GO,NO> > AOP(new AlgebraicOverlappingPreconditioner<SC,LO,GO,NO>(A,paramList_));
#else
                RCP<AlgebraicOverlappingPreconditioner<SC,NO> > AOP(new AlgebraicOverlappingPreconditioner<SC,NO>(A,paramList_));
#endif
                
                AOP->initialize(paramList_->get("Overlap",1),
                                repeatedMap);
                
                SchwarzPreconditioner = AOP;
            } else if (!paramList_->get("FROSch Preconditioner Type","TwoLevelPreconditioner").compare("GDSWPreconditioner")) {
                // Extract the repeated map
                ConstXMapPtr repeatedMap = extractRepeatedMap(comm,underlyingLib);
                
                // Extract the coordinate list
                ConstXMultiVectorPtr coordinatesList = extractCoordinatesList(comm,underlyingLib);
                
                // Extract the dof ordering
                DofOrdering dofOrdering = NodeWise;
                if (!paramList_->get("DofOrdering","NodeWise").compare("NodeWise")) {
                    dofOrdering = NodeWise;
                } else if (!paramList_->get("DofOrdering","NodeWise").compare("DimensionWise")) {
                    dofOrdering = DimensionWise;
                } else if (!paramList_->get("DofOrdering","NodeWise").compare("Custom")) {
                    dofOrdering = Custom;
                } else {
                    FROSCH_ASSERT(false,"ERROR: Specify a valid DofOrdering.");
                }
                
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                RCP<GDSWPreconditioner<SC,LO,GO,NO> > GP(new GDSWPreconditioner<SC,LO,GO,NO>(A,paramList_));
#else
                RCP<GDSWPreconditioner<SC,NO> > GP(new GDSWPreconditioner<SC,NO>(A,paramList_));
#endif
                
                GP->initialize(paramList_->get("Dimension",3),
                               paramList_->get("DofsPerNode",1),
                               dofOrdering,
                               paramList_->get("Overlap",1),
                               repeatedMap,
                               coordinatesList);
                
                SchwarzPreconditioner = GP;
            } else if (!paramList_->get("FROSch Preconditioner Type","TwoLevelPreconditioner").compare("RGDSWPreconditioner")) {
                // Extract the repeated map
                ConstXMapPtr repeatedMap = extractRepeatedMap(comm,underlyingLib);
                
                // Extract the coordinate list
                ConstXMultiVectorPtr coordinatesList = extractCoordinatesList(comm,underlyingLib);
                
                // Extract the dof ordering
                DofOrdering dofOrdering = NodeWise;
                if (!paramList_->get("DofOrdering","NodeWise").compare("NodeWise")) {
                    dofOrdering = NodeWise;
                } else if (!paramList_->get("DofOrdering","NodeWise").compare("DimensionWise")) {
                    dofOrdering = DimensionWise;
                } else if (!paramList_->get("DofOrdering","NodeWise").compare("Custom")) {
                    dofOrdering = Custom;
                } else {
                    FROSCH_ASSERT(false,"ERROR: Specify a valid DofOrdering.");
                }
                
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                RCP<RGDSWPreconditioner<SC,LO,GO,NO> > RGP(new RGDSWPreconditioner<SC,LO,GO,NO>(A,paramList_));
#else
                RCP<RGDSWPreconditioner<SC,NO> > RGP(new RGDSWPreconditioner<SC,NO>(A,paramList_));
#endif
                
                RGP->initialize(paramList_->get("Dimension",3),
                                paramList_->get("DofsPerNode",1),
                                dofOrdering,
                                paramList_->get("Overlap",1),
                                repeatedMap,
                                coordinatesList);
                
                SchwarzPreconditioner = RGP;
            } else if (!paramList_->get("FROSch Preconditioner Type","TwoLevelPreconditioner").compare("OneLevelPreconditioner")) {
                // Extract the repeated map
                ConstXMapPtr repeatedMap = extractRepeatedMap(comm,underlyingLib);
                
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                RCP<OneLevelPreconditioner<SC,LO,GO,NO> > OLP(new OneLevelPreconditioner<SC,LO,GO,NO>(A,paramList_));
#else
                RCP<OneLevelPreconditioner<SC,NO> > OLP(new OneLevelPreconditioner<SC,NO>(A,paramList_));
#endif
                
                OLP->initialize(paramList_->get("Overlap",1),
                                repeatedMap);
                
                SchwarzPreconditioner = OLP;
            } else if (!paramList_->get("FROSch Preconditioner Type","TwoLevelPreconditioner").compare("TwoLevelPreconditioner")) {
                // Extract the repeated map
                ConstXMapPtr repeatedMap = extractRepeatedMap(comm,underlyingLib);
                
                // Extract the null space
                ConstXMultiVectorPtr nullSpaceBasis = extractNullSpace(comm,underlyingLib);
                
                // Extract the coordinate list
                ConstXMultiVectorPtr coordinatesList = extractCoordinatesList(comm,underlyingLib);
                
                // Extract the dof ordering
                DofOrdering dofOrdering = NodeWise;
                if (!paramList_->get("DofOrdering","NodeWise").compare("NodeWise")) {
                    dofOrdering = NodeWise;
                } else if (!paramList_->get("DofOrdering","NodeWise").compare("DimensionWise")) {
                    dofOrdering = DimensionWise;
                } else if (!paramList_->get("DofOrdering","NodeWise").compare("Custom")) {
                    dofOrdering = Custom;
                } else {
                    FROSCH_ASSERT(false,"ERROR: Specify a valid DofOrdering.");
                }
                
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                RCP<TwoLevelPreconditioner<SC,LO,GO,NO> > TLP(new TwoLevelPreconditioner<SC,LO,GO,NO>(A,paramList_));
#else
                RCP<TwoLevelPreconditioner<SC,NO> > TLP(new TwoLevelPreconditioner<SC,NO>(A,paramList_));
#endif
                
                TLP->initialize(paramList_->get("Dimension",3),
                                paramList_->get("DofsPerNode",1),
                                paramList_->get("Overlap",1),
                                nullSpaceBasis,
                                coordinatesList,
                                dofOrdering,
                                repeatedMap);
                
                SchwarzPreconditioner = TLP;
            } else if (!paramList_->get("FROSch Preconditioner Type","TwoLevelPreconditioner").compare("TwoLevelBlockPreconditioner")) {
                ConstXMapPtrVecPtr repeatedMaps = null;
                UNVecPtr dofsPerNodeVector;
                DofOrderingVecPtr dofOrderings;
                
                FROSCH_ASSERT(paramList_->isParameter("DofsPerNode Vector"),"Currently, TwoLevelBlockPreconditioner cannot be constructed without DofsPerNode Vector.");
                FROSCH_ASSERT(paramList_->isParameter("DofOrdering Vector"),"Currently, TwoLevelBlockPreconditioner cannot be constructed without DofOrdering Vector.");
                if (paramList_->isParameter("Repeated Map Vector")) {
                    XMapPtrVecPtr repeatedMapsTmp = ExtractVectorFromParameterList<XMapPtr>(*paramList_,"Repeated Map Vector");
                    if (!repeatedMapsTmp.is_null()) {
                        repeatedMaps.resize(repeatedMapsTmp.size());
                        for (unsigned i=0; i<repeatedMaps.size(); i++) {
                            repeatedMaps[i] = repeatedMapsTmp[i].getConst();
                        }
                    }
                    FROSCH_ASSERT(!repeatedMaps.is_null(),"FROSch::FROSchFactory : ERROR: repeatedMaps.is_null()");
                    
                    dofsPerNodeVector = ExtractVectorFromParameterList<UN>(*paramList_,"DofsPerNode Vector");
                    dofOrderings = ExtractVectorFromParameterList<DofOrdering>(*paramList_,"DofOrdering Vector");
                } else {
                    FROSCH_ASSERT(false,"Currently, TwoLevelBlockPreconditioner cannot be constructed without Repeated Maps.");
                }
                
                FROSCH_ASSERT(repeatedMaps.size()==dofsPerNodeVector.size(),"RepeatedMaps.size()!=dofsPerNodeVector.size()");
                FROSCH_ASSERT(repeatedMaps.size()==dofOrderings.size(),"RepeatedMaps.size()!=dofOrderings.size()");
                
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                RCP<TwoLevelBlockPreconditioner<SC,LO,GO,NO> > TLBP(new TwoLevelBlockPreconditioner<SC,LO,GO,NO>(A,paramList_));
#else
                RCP<TwoLevelBlockPreconditioner<SC,NO> > TLBP(new TwoLevelBlockPreconditioner<SC,NO>(A,paramList_));
#endif
                
                TLBP->initialize(paramList_->get("Dimension",3),
                                 dofsPerNodeVector,
                                 dofOrderings,
                                 paramList_->get("Overlap",1),
                                 repeatedMaps);
                
                SchwarzPreconditioner = TLBP;
            } else {
                FROSCH_ASSERT(false,"Thyra::FROSchFactory : ERROR: Preconditioner Type is unknown.");
            }
            
            SchwarzPreconditioner->compute();
            //-----------------------------------------------
            
            LinearOpBasePtr thyraPrecOp = null;
            //FROSCh_XpetraOP
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            ConstVectorSpaceBasePtr thyraRangeSpace  = ThyraUtils<SC,LO,GO,NO>::toThyra(SchwarzPreconditioner->getRangeMap());
            ConstVectorSpaceBasePtr thyraDomainSpace = ThyraUtils<SC,LO,GO,NO>::toThyra(SchwarzPreconditioner->getDomainMap());
#else
            ConstVectorSpaceBasePtr thyraRangeSpace  = ThyraUtils<SC,NO>::toThyra(SchwarzPreconditioner->getRangeMap());
            ConstVectorSpaceBasePtr thyraDomainSpace = ThyraUtils<SC,NO>::toThyra(SchwarzPreconditioner->getDomainMap());
#endif
            
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            RCP<Operator<SC,LO,GO,NO> > xpOp = rcp_dynamic_cast<Operator<SC,LO,GO,NO> >(SchwarzPreconditioner);
#else
            RCP<Operator<SC,NO> > xpOp = rcp_dynamic_cast<Operator<SC,NO> >(SchwarzPreconditioner);
#endif
            
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            thyraPrecOp = fROSchLinearOp<SC,LO,GO,NO>(thyraRangeSpace,thyraDomainSpace,xpOp,bIsEpetra,bIsTpetra);
#else
            thyraPrecOp = fROSchLinearOp<SC,NO>(thyraRangeSpace,thyraDomainSpace,xpOp,bIsEpetra,bIsTpetra);
#endif
            
            TEUCHOS_TEST_FOR_EXCEPT(is_null(thyraPrecOp));
            
            //Set SchwarzPreconditioner
            defaultPrec->initializeUnspecified(thyraPrecOp);            
        } else {
            // cast to SchwarzPreconditioner
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            RCP<FROSchLinearOp<SC,LO,GO,NO> > fROSch_LinearOp = rcp_dynamic_cast<FROSchLinearOp<SC,LO,GO,NO> >(thyra_precOp,true);
            RCP<Operator<SC,LO,GO,NO> > xpetraOp = fROSch_LinearOp->getXpetraOperator();
#else
            RCP<FROSchLinearOp<SC,NO> > fROSch_LinearOp = rcp_dynamic_cast<FROSchLinearOp<SC,NO> >(thyra_precOp,true);
            RCP<Operator<SC,NO> > xpetraOp = fROSch_LinearOp->getXpetraOperator();
#endif
            
            if (!paramList_->get("FROSch Preconditioner Type","TwoLevelPreconditioner").compare("AlgebraicOverlappingPreconditioner")) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                RCP<AlgebraicOverlappingPreconditioner<SC,LO,GO,NO> > AOP = rcp_dynamic_cast<AlgebraicOverlappingPreconditioner<SC,LO,GO,NO> >(xpetraOp, true);
#else
                RCP<AlgebraicOverlappingPreconditioner<SC,NO> > AOP = rcp_dynamic_cast<AlgebraicOverlappingPreconditioner<SC,NO> >(xpetraOp, true);
#endif
                AOP->resetMatrix(A);
                SchwarzPreconditioner = AOP;
            } else if (!paramList_->get("FROSch Preconditioner Type","TwoLevelPreconditioner").compare("GDSWPreconditioner")) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                RCP<AlgebraicOverlappingPreconditioner<SC,LO,GO,NO> > GP = rcp_dynamic_cast<AlgebraicOverlappingPreconditioner<SC,LO,GO,NO> >(xpetraOp, true);
#else
                RCP<AlgebraicOverlappingPreconditioner<SC,NO> > GP = rcp_dynamic_cast<AlgebraicOverlappingPreconditioner<SC,NO> >(xpetraOp, true);
#endif
                GP->resetMatrix(A);
                SchwarzPreconditioner = GP;
            } else if (!paramList_->get("FROSch Preconditioner Type","TwoLevelPreconditioner").compare("RGDSWPreconditioner")) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                RCP<OneLevelPreconditioner<SC,LO,GO,NO> > RGP = rcp_dynamic_cast<OneLevelPreconditioner<SC,LO,GO,NO> >(xpetraOp, true);
#else
                RCP<OneLevelPreconditioner<SC,NO> > RGP = rcp_dynamic_cast<OneLevelPreconditioner<SC,NO> >(xpetraOp, true);
#endif
                RGP->resetMatrix(A);
                SchwarzPreconditioner = RGP;
            } else if (!paramList_->get("FROSch Preconditioner Type","TwoLevelPreconditioner").compare("OneLevelPreconditioner")) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                RCP<OneLevelPreconditioner<SC,LO,GO,NO> > OLP = rcp_dynamic_cast<OneLevelPreconditioner<SC,LO,GO,NO> >(xpetraOp, true);
#else
                RCP<OneLevelPreconditioner<SC,NO> > OLP = rcp_dynamic_cast<OneLevelPreconditioner<SC,NO> >(xpetraOp, true);
#endif
                OLP->resetMatrix(A);
                SchwarzPreconditioner = OLP;
            } else if (!paramList_->get("FROSch Preconditioner Type","TwoLevelPreconditioner").compare("TwoLevelPreconditioner")) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                RCP<OneLevelPreconditioner<SC,LO,GO,NO> > TLP = rcp_dynamic_cast<OneLevelPreconditioner<SC,LO,GO,NO> >(xpetraOp, true);
#else
                RCP<OneLevelPreconditioner<SC,NO> > TLP = rcp_dynamic_cast<OneLevelPreconditioner<SC,NO> >(xpetraOp, true);
#endif
                TLP->resetMatrix(A);
                SchwarzPreconditioner = TLP;
            } else if (!paramList_->get("FROSch Preconditioner Type","TwoLevelPreconditioner").compare("TwoLevelBlockPreconditioner")) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                RCP<OneLevelPreconditioner<SC,LO,GO,NO> > TLBP = rcp_dynamic_cast<OneLevelPreconditioner<SC,LO,GO,NO> >(xpetraOp, true);
#else
                RCP<OneLevelPreconditioner<SC,NO> > TLBP = rcp_dynamic_cast<OneLevelPreconditioner<SC,NO> >(xpetraOp, true);
#endif
                TLBP->resetMatrix(A);
                SchwarzPreconditioner = TLBP;
            } else {
                FROSCH_ASSERT(false,"Thyra::FROSchFactory : ERROR: Preconditioner Type is unknown.");
            }
            // recompute SchwarzPreconditioner
            SchwarzPreconditioner->compute();
        }
    }

    //-------------------------------------------------------------
    //uninitialize
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    void FROSchFactory<SC,LO,GO,NO>::uninitializePrec(PreconditionerBase<SC>* prec,
#else
    template <class SC, class NO>
    void FROSchFactory<SC,NO>::uninitializePrec(PreconditionerBase<SC>* prec,
#endif
                                                      ConstLinearOpSourceBasePtr* fwdOp,
                                                      ESupportSolveUse* supportSolveUse) const
    {
        TEUCHOS_ASSERT(prec);

        // Retrieve concrete preconditioner object
        const Ptr<DefaultPreconditioner<SC> > defaultPrec = ptr(dynamic_cast<DefaultPreconditioner<SC> *>(prec));
        TEUCHOS_TEST_FOR_EXCEPT(is_null(defaultPrec));

        if (fwdOp) {
            // TODO: Implement properly instead of returning default value
            *fwdOp = null;
        }

        if (supportSolveUse) {
            // TODO: Implement properly instead of returning default value
            *supportSolveUse = SUPPORT_SOLVE_UNSPECIFIED;
        }

        defaultPrec->uninitialize();
    }
    //-----------------------------------------------------------------
    //Following Functione maybe needed later
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    void FROSchFactory<SC,LO,GO,NO>::setParameterList(ParameterListPtr const & paramList)
#else
    template <class SC, class NO>
    void FROSchFactory<SC,NO>::setParameterList(ParameterListPtr const & paramList)
#endif
    {
        TEUCHOS_TEST_FOR_EXCEPT(is_null(paramList));
        paramList_ = paramList;
    }
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC, class LO,class GO, class NO>
    typename FROSchFactory<SC,LO,GO,NO>::ParameterListPtr FROSchFactory<SC,LO,GO,NO>::unsetParameterList()
#else
    template<class SC, class NO>
    typename FROSchFactory<SC,NO>::ParameterListPtr FROSchFactory<SC,NO>::unsetParameterList()
#endif
    {
        ParameterListPtr savedParamList = paramList_;
        paramList_ = null;
        return savedParamList;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    typename FROSchFactory<SC,LO,GO,NO>::ParameterListPtr FROSchFactory<SC,LO,GO,NO>::getNonconstParameterList()
#else
    template <class SC, class NO>
    typename FROSchFactory<SC,NO>::ParameterListPtr FROSchFactory<SC,NO>::getNonconstParameterList()
#endif
    {
        return paramList_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    typename FROSchFactory<SC,LO,GO,NO>::ConstParameterListPtr FROSchFactory<SC,LO,GO,NO>::getParameterList() const
#else
    template <class SC, class NO>
    typename FROSchFactory<SC,NO>::ConstParameterListPtr FROSchFactory<SC,NO>::getParameterList() const
#endif
    {
        return paramList_;
    }
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    typename FROSchFactory<SC,LO,GO,NO>::ConstParameterListPtr FROSchFactory<SC,LO,GO,NO>::getValidParameters() const
#else
    template <class SC, class NO>
    typename FROSchFactory<SC,NO>::ConstParameterListPtr FROSchFactory<SC,NO>::getValidParameters() const
#endif
    {
        static ConstParameterListPtr validPL;

        if (is_null(validPL))
        validPL = rcp(new ParameterList());

        return validPL;
    }
   
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    std::string FROSchFactory<SC,LO,GO,NO>::description() const
#else
    template <class SC, class NO>
    std::string FROSchFactory<SC,NO>::description() const
#endif
    {
        return "FROSchFactory";
    }
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    typename FROSchFactory<SC,LO,GO,NO>::ConstXMapPtr FROSchFactory<SC,LO,GO,NO>::extractRepeatedMap(CommPtr comm,
#else
    template <class SC, class NO>
    typename FROSchFactory<SC,NO>::ConstXMapPtr FROSchFactory<SC,NO>::extractRepeatedMap(CommPtr comm,
#endif
                                                                                                     UnderlyingLib lib) const
    {
        ConstXMapPtr repeatedMap = null;
        if (paramList_->isParameter("Repeated Map")) {
            repeatedMap = ExtractPtrFromParameterList<XMap>(*paramList_,"Repeated Map").getConst();
            if (repeatedMap.is_null()) {
                if (lib==UseTpetra) { // If coordinatesList.is_null(), we look for Tpetra/Epetra RCPs
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                    RCP<const Tpetra::Map<LO,GO,NO> > repeatedMapTmp = ExtractPtrFromParameterList<const Tpetra::Map<LO,GO,NO> >(*paramList_,"Repeated Map");
#else
                    RCP<const Tpetra::Map<NO> > repeatedMapTmp = ExtractPtrFromParameterList<const Tpetra::Map<NO> >(*paramList_,"Repeated Map");
#endif
                    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                    RCP<const TpetraMap<LO,GO,NO> > xTpetraRepeatedMap(new const TpetraMap<LO,GO,NO>(repeatedMapTmp));
#else
                    RCP<const TpetraMap<NO> > xTpetraRepeatedMap(new const TpetraMap<NO>(repeatedMapTmp));
#endif
                    repeatedMap = rcp_dynamic_cast<ConstXMap>(xTpetraRepeatedMap);
                } else {
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
                    FROSCH_WARNING("FROSch::FROSchFactory",comm->getRank()==0,"Cannot retrieve Epetra objects from ParameterList. Use Xpetra instead.");
#endif
                }
            }
            FROSCH_ASSERT(!repeatedMap.is_null(),"FROSch::FROSchFactory : ERROR: repeatedMap.is_null()");
        }
        return repeatedMap;
    }
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    typename FROSchFactory<SC,LO,GO,NO>::ConstXMultiVectorPtr FROSchFactory<SC,LO,GO,NO>::extractCoordinatesList(CommPtr comm,
#else
    template <class SC, class NO>
    typename FROSchFactory<SC,NO>::ConstXMultiVectorPtr FROSchFactory<SC,NO>::extractCoordinatesList(CommPtr comm,
#endif
                                                                                                                 UnderlyingLib lib) const
    {
        ConstXMultiVectorPtr coordinatesList = null;
        if (paramList_->isParameter("Coordinates List")) {
            coordinatesList = ExtractPtrFromParameterList<XMultiVector>(*paramList_,"Coordinates List").getConst();
            if (coordinatesList.is_null()) {
                if (lib==UseTpetra) { // If coordinatesList.is_null(), we look for Tpetra/Epetra RCPs
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                    RCP<Tpetra::MultiVector<SC,LO,GO,NO> > coordinatesListTmp = ExtractPtrFromParameterList<Tpetra::MultiVector<SC,LO,GO,NO> >(*paramList_,"Coordinates List");
#else
                    RCP<Tpetra::MultiVector<SC,NO> > coordinatesListTmp = ExtractPtrFromParameterList<Tpetra::MultiVector<SC,NO> >(*paramList_,"Coordinates List");
#endif
                    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                    RCP<const Xpetra::TpetraMultiVector<SC,LO,GO,NO> > xTpetraCoordinatesList(new const Xpetra::TpetraMultiVector<SC,LO,GO,NO>(coordinatesListTmp));
#else
                    RCP<const Xpetra::TpetraMultiVector<SC,NO> > xTpetraCoordinatesList(new const Xpetra::TpetraMultiVector<SC,NO>(coordinatesListTmp));
#endif
                    coordinatesList = rcp_dynamic_cast<ConstXMultiVector>(xTpetraCoordinatesList);
                } else {
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
                    FROSCH_WARNING("FROSch::FROSchFactory",comm->getRank()==0,"Cannot retrieve Epetra objects from ParameterList. Use Xpetra instead.");
#endif
                }
            }
            FROSCH_ASSERT(!coordinatesList.is_null(),"FROSch::FROSchFactory : ERROR: coordinatesList.is_null()");
        }
        return coordinatesList;
    }
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    typename FROSchFactory<SC,LO,GO,NO>::ConstXMultiVectorPtr FROSchFactory<SC,LO,GO,NO>::extractNullSpace(CommPtr comm,
#else
    template <class SC, class NO>
    typename FROSchFactory<SC,NO>::ConstXMultiVectorPtr FROSchFactory<SC,NO>::extractNullSpace(CommPtr comm,
#endif
                                                                                                           UnderlyingLib lib) const
    {
        ConstXMultiVectorPtr nullSpaceBasis = null;
        if (paramList_->isParameter("Null Space")) {
            nullSpaceBasis = ExtractPtrFromParameterList<XMultiVector>(*paramList_,"Null Space").getConst();
            if (nullSpaceBasis.is_null()) {
                if (lib==UseTpetra) { // If nullSpaceBasis.is_null(), we look for Tpetra/Epetra RCPs
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                    RCP<Tpetra::MultiVector<SC,LO,GO,NO> > nullSpaceBasisTmp = ExtractPtrFromParameterList<Tpetra::MultiVector<SC,LO,GO,NO> >(*paramList_,"Null Space");
#else
                    RCP<Tpetra::MultiVector<SC,NO> > nullSpaceBasisTmp = ExtractPtrFromParameterList<Tpetra::MultiVector<SC,NO> >(*paramList_,"Null Space");
#endif
                    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                    RCP<const Xpetra::TpetraMultiVector<SC,LO,GO,NO> > xTpetraNullSpaceBasis(new const Xpetra::TpetraMultiVector<SC,LO,GO,NO>(nullSpaceBasisTmp));
#else
                    RCP<const Xpetra::TpetraMultiVector<SC,NO> > xTpetraNullSpaceBasis(new const Xpetra::TpetraMultiVector<SC,NO>(nullSpaceBasisTmp));
#endif
                    nullSpaceBasis = rcp_dynamic_cast<ConstXMultiVector>(xTpetraNullSpaceBasis);
                } else {
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
                    FROSCH_WARNING("FROSch::FROSchFactory",comm->getRank()==0,"Cannot retrieve Epetra objects from ParameterList. Use Xpetra instead.");
#endif
                }
            }
            FROSCH_ASSERT(!nullSpaceBasis.is_null(),"FROSch::FROSchFactory : ERROR: nullSpaceBasis.is_null()");
        }
        return nullSpaceBasis;
    }

}
#endif


