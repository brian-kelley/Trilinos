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

#ifndef _FROSCH_ALGEBRAICOVERLAPPINGPRECONDITIONER_DEF_HPP
#define _FROSCH_ALGEBRAICOVERLAPPINGPRECONDITIONER_DEF_HPP

#include <FROSch_AlgebraicOverlappingPreconditioner_decl.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    AlgebraicOverlappingPreconditioner<SC,LO,GO,NO>::AlgebraicOverlappingPreconditioner(ConstXMatrixPtr k,
#else
    template <class SC,class NO>
    AlgebraicOverlappingPreconditioner<SC,NO>::AlgebraicOverlappingPreconditioner(ConstXMatrixPtr k,
#endif
                                                                                        ParameterListPtr parameterList) :
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    SchwarzPreconditioner<SC,LO,GO,NO> (parameterList,k->getRangeMap()->getComm()),
#else
    SchwarzPreconditioner<SC,NO> (parameterList,k->getRangeMap()->getComm()),
#endif
    K_ (k),
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    SumOperator_ (new SumOperator<SC,LO,GO,NO>(k->getRangeMap()->getComm()))
#else
    SumOperator_ (new SumOperator<SC,NO>(k->getRangeMap()->getComm()))
#endif
    {
        FROSCH_TIMER_START_LEVELID(algebraicOverlappingPreconditionerTime,"AlgebraicOverlappingPreconditioner::AlgebraicOverlappingPreconditioner");
        // Set the LevelID in the sublist
        parameterList->sublist("AlgebraicOverlappingOperator").set("Level ID",this->LevelID_);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        OverlappingOperator_.reset(new AlgebraicOverlappingOperator<SC,LO,GO,NO>(k,sublist(parameterList,"AlgebraicOverlappingOperator")));
#else
        OverlappingOperator_.reset(new AlgebraicOverlappingOperator<SC,NO>(k,sublist(parameterList,"AlgebraicOverlappingOperator")));
#endif
        SumOperator_->addOperator(OverlappingOperator_);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingPreconditioner<SC,LO,GO,NO>::initialize(bool useDefaultParameters)
#else
    template <class SC,class NO>
    int AlgebraicOverlappingPreconditioner<SC,NO>::initialize(bool useDefaultParameters)
#endif
    {
        return initialize(1,null);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingPreconditioner<SC,LO,GO,NO>::initialize(int overlap,
#else
    template <class SC,class NO>
    int AlgebraicOverlappingPreconditioner<SC,NO>::initialize(int overlap,
#endif
                                                                    ConstXMapPtr repeatedMap)
    {
        FROSCH_TIMER_START_LEVELID(initializeTime,"AlgebraicOverlappingPreconditioner::initialize");
        return OverlappingOperator_->initialize(overlap,repeatedMap);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingPreconditioner<SC,LO,GO,NO>::compute()
#else
    template <class SC,class NO>
    int AlgebraicOverlappingPreconditioner<SC,NO>::compute()
#endif
    {
        FROSCH_TIMER_START_LEVELID(computeTime,"AlgebraicOverlappingPreconditioner::compute");
        return OverlappingOperator_->compute();
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    void AlgebraicOverlappingPreconditioner<SC,LO,GO,NO>::apply(const XMultiVector &x,
#else
    template <class SC,class NO>
    void AlgebraicOverlappingPreconditioner<SC,NO>::apply(const XMultiVector &x,
#endif
                                                                XMultiVector &y,
                                                                ETransp mode,
                                                                SC alpha,
                                                                SC beta) const
    {
        FROSCH_TIMER_START_LEVELID(applyTime,"AlgebraicOverlappingPreconditioner::apply");
        return SumOperator_->apply(x,y,true,mode,alpha,beta);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    typename AlgebraicOverlappingPreconditioner<SC,LO,GO,NO>::ConstXMapPtr AlgebraicOverlappingPreconditioner<SC,LO,GO,NO>::getDomainMap() const
#else
    template <class SC,class NO>
    typename AlgebraicOverlappingPreconditioner<SC,NO>::ConstXMapPtr AlgebraicOverlappingPreconditioner<SC,NO>::getDomainMap() const
#endif
    {
        return K_->getDomainMap();
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    typename AlgebraicOverlappingPreconditioner<SC,LO,GO,NO>::ConstXMapPtr AlgebraicOverlappingPreconditioner<SC,LO,GO,NO>::getRangeMap() const
#else
    template <class SC,class NO>
    typename AlgebraicOverlappingPreconditioner<SC,NO>::ConstXMapPtr AlgebraicOverlappingPreconditioner<SC,NO>::getRangeMap() const
#endif
    {
        return K_->getRangeMap();
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    void AlgebraicOverlappingPreconditioner<SC,LO,GO,NO>::describe(FancyOStream &out,
#else
    template <class SC,class NO>
    void AlgebraicOverlappingPreconditioner<SC,NO>::describe(FancyOStream &out,
#endif
                                                                   const EVerbosityLevel verbLevel) const
    {
        SumOperator_->describe(out,verbLevel);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    std::string AlgebraicOverlappingPreconditioner<SC,LO,GO,NO>::description() const
#else
    template <class SC,class NO>
    std::string AlgebraicOverlappingPreconditioner<SC,NO>::description() const
#endif
    {
        return "Algebraic Overlapping Preconditioner";
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingPreconditioner<SC,LO,GO,NO>::resetMatrix(ConstXMatrixPtr &k)
#else
    template <class SC,class NO>
    int AlgebraicOverlappingPreconditioner<SC,NO>::resetMatrix(ConstXMatrixPtr &k)
#endif
    {
        FROSCH_TIMER_START_LEVELID(resetMatrixTime,"TwoLevelPreconditioner::resetMatrix");
        this->K_ = k;
        OverlappingOperator_->resetMatrix(this->K_);
        return 0;
    }
}

#endif
