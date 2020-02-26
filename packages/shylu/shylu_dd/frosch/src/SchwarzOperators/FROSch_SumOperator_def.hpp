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

#ifndef _FROSCH_SUMOPERATOR_DEF_HPP
#define _FROSCH_SUMOPERATOR_DEF_HPP

#include <FROSch_SumOperator_decl.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    SumOperator<SC,LO,GO,NO>::SumOperator(CommPtr comm) :
    SchwarzOperator<SC,LO,GO,NO> (comm)
#else
    template <class SC,class NO>
    SumOperator<SC,NO>::SumOperator(CommPtr comm) :
    SchwarzOperator<SC,NO> (comm)
#endif
    {
        FROSCH_TIMER_START_LEVELID(sumOperatorTime,"SumOperator::SumOperator");
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    SumOperator<SC,LO,GO,NO>::SumOperator(SchwarzOperatorPtrVecPtr operators) :
    SchwarzOperator<SC,LO,GO,NO> (operators[0]->getRangeMap()->getComm())
#else
    template <class SC,class NO>
    SumOperator<SC,NO>::SumOperator(SchwarzOperatorPtrVecPtr operators) :
    SchwarzOperator<SC,NO> (operators[0]->getRangeMap()->getComm())
#endif
    {
        FROSCH_TIMER_START_LEVELID(sumOperatorTime,"SumOperator::SumOperator");
        FROSCH_ASSERT(operators.size()>0,"operators.size()<=0");
        OperatorVector_.push_back(operators[0]);
        for (unsigned i=1; i<operators.size(); i++) {
            FROSCH_ASSERT(operators[i]->OperatorDomainMap().SameAs(OperatorVector_[0]->OperatorDomainMap()),"The DomainMaps of the operators are not identical.");
            FROSCH_ASSERT(operators[i]->OperatorRangeMap().SameAs(OperatorVector_[0]->OperatorRangeMap()),"The RangeMaps of the operators are not identical.");

            OperatorVector_.push_back(operators[i]);
            EnableOperators_.push_back(true);
        }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    SumOperator<SC,LO,GO,NO>::~SumOperator()
#else
    template <class SC,class NO>
    SumOperator<SC,NO>::~SumOperator()
#endif
    {

    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int SumOperator<SC,LO,GO,NO>::initialize()
#else
    template <class SC,class NO>
    int SumOperator<SC,NO>::initialize()
#endif
    {
        if (this->Verbose_) {
            FROSCH_ASSERT(false,"ERROR: Each of the Operators has to be initialized manually.");
        }
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int SumOperator<SC,LO,GO,NO>::initialize(ConstXMapPtr repeatedMap)
#else
    template <class SC,class NO>
    int SumOperator<SC,NO>::initialize(ConstXMapPtr repeatedMap)
#endif
    {
        if (this->Verbose_) {
            FROSCH_ASSERT(false,"ERROR: Each of the Operators has to be initialized manually.");
        }
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int SumOperator<SC,LO,GO,NO>::compute()
#else
    template <class SC,class NO>
    int SumOperator<SC,NO>::compute()
#endif
    {
        if (this->Verbose_) {
            FROSCH_ASSERT(false,"ERROR: Each of the Operators has to be computed manually.");
        }
        return 0;
    }

    // Y = alpha * A^mode * X + beta * Y
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    void SumOperator<SC,LO,GO,NO>::apply(const XMultiVector &x,
#else
    template <class SC,class NO>
    void SumOperator<SC,NO>::apply(const XMultiVector &x,
#endif
                                         XMultiVector &y,
                                         bool usePreconditionerOnly,
                                         ETransp mode,
                                         SC alpha,
                                         SC beta) const
    {
        FROSCH_TIMER_START_LEVELID(applyTime,"SumOperator::apply");
        if (OperatorVector_.size()>0) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            if (XTmp_.is_null()) XTmp_ = MultiVectorFactory<SC,LO,GO,NO>::Build(x.getMap(),x.getNumVectors());
#else
            if (XTmp_.is_null()) XTmp_ = MultiVectorFactory<SC,NO>::Build(x.getMap(),x.getNumVectors());
#endif
            *XTmp_ = x; // Das brauche ich für den Fall das x=y
            UN itmp = 0;
            for (UN i=0; i<OperatorVector_.size(); i++) {
                if (EnableOperators_[i]) {
                    OperatorVector_[i]->apply(*XTmp_,y,usePreconditionerOnly,mode,alpha,beta);
                    if (itmp==0) beta = ScalarTraits<SC>::one();
                    itmp++;
                }
            }
        } else {
            y.update(alpha,x,beta);
        }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    typename SumOperator<SC,LO,GO,NO>::ConstXMapPtr SumOperator<SC,LO,GO,NO>::getDomainMap() const
#else
    template <class SC,class NO>
    typename SumOperator<SC,NO>::ConstXMapPtr SumOperator<SC,NO>::getDomainMap() const
#endif
    {
        return OperatorVector_[0]->getDomainMap();
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    typename SumOperator<SC,LO,GO,NO>::ConstXMapPtr SumOperator<SC,LO,GO,NO>::getRangeMap() const
#else
    template <class SC,class NO>
    typename SumOperator<SC,NO>::ConstXMapPtr SumOperator<SC,NO>::getRangeMap() const
#endif
    {
        return OperatorVector_[0]->getRangeMap();
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    void SumOperator<SC,LO,GO,NO>::describe(FancyOStream &out,
#else
    template <class SC,class NO>
    void SumOperator<SC,NO>::describe(FancyOStream &out,
#endif
                                            const EVerbosityLevel verbLevel) const
    {
        FROSCH_ASSERT(false,"describe() has to be implemented properly...");
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    std::string SumOperator<SC,LO,GO,NO>::description() const
#else
    template <class SC,class NO>
    std::string SumOperator<SC,NO>::description() const
#endif
    {
        std::string labelString = "Sum operator: ";

        for (UN i=0; i<OperatorVector_.size(); i++) {
            labelString += OperatorVector_[i]->description();
            if (i<OperatorVector_.size()-1) {
                labelString += ",";
            }
        }
        return labelString;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int SumOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr op)
#else
    template <class SC,class NO>
    int SumOperator<SC,NO>::addOperator(SchwarzOperatorPtr op)
#endif
    {
        FROSCH_TIMER_START_LEVELID(addOperatorTime,"SumOperator::addOperator");
        int ret = 0;
        if (OperatorVector_.size()>0) {
            if (!op->getDomainMap()->isSameAs(*OperatorVector_[0]->getDomainMap())) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                if (this->Verbose_) std::cerr << "SumOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr op)\t\t!op->getDomainMap().isSameAs(OperatorVector_[0]->getDomainMap())\n";
#else
                if (this->Verbose_) std::cerr << "SumOperator<SC,NO>::addOperator(SchwarzOperatorPtr op)\t\t!op->getDomainMap().isSameAs(OperatorVector_[0]->getDomainMap())\n";
#endif
                ret -= 1;
            }
            if (!op->getRangeMap()->isSameAs(*OperatorVector_[0]->getRangeMap())){
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                if (this->Verbose_) std::cerr << "SumOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr op)\t\t!op->getRangeMap().isSameAs(OperatorVector_[0]->getRangeMap())\n";
#else
                if (this->Verbose_) std::cerr << "SumOperator<SC,NO>::addOperator(SchwarzOperatorPtr op)\t\t!op->getRangeMap().isSameAs(OperatorVector_[0]->getRangeMap())\n";
#endif
                ret -= 10;
            }
            //FROSCH_ASSERT(op->OperatorDomainMap().SameAs(OperatorVector_[0]->OperatorDomainMap()),"The DomainMaps of the operators are not identical.");
            //FROSCH_ASSERT(op->OperatorRangeMap().SameAs(OperatorVector_[0]->OperatorRangeMap()),"The RangeMaps of the operators are not identical.");
        }
        OperatorVector_.push_back(op);
        EnableOperators_.push_back(true);
        return ret;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int SumOperator<SC,LO,GO,NO>::addOperators(SchwarzOperatorPtrVecPtr operators)
#else
    template <class SC,class NO>
    int SumOperator<SC,NO>::addOperators(SchwarzOperatorPtrVecPtr operators)
#endif
    {
        FROSCH_TIMER_START_LEVELID(addOperatorsTime,"SumOperator::addOperators");
        int ret = 0;
        for (UN i=1; i<operators.size(); i++) {
            if (0>addOperator(operators[i])) ret -= pow(10,i);
        }
        return ret;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int SumOperator<SC,LO,GO,NO>::resetOperator(UN iD,
#else
    template <class SC,class NO>
    int SumOperator<SC,NO>::resetOperator(UN iD,
#endif
                                                SchwarzOperatorPtr op)
    {
        FROSCH_TIMER_START_LEVELID(resetOperatorTime,"SumOperator::resetOperator");
        FROSCH_ASSERT(iD<OperatorVector_.size(),"iD exceeds the length of the OperatorVector_");
        int ret = 0;
        if (!op->getDomainMap().isSameAs(OperatorVector_[0]->getDomainMap())) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            if (this->Verbose_) std::cerr << "SumOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr op)\t\t!op->getDomainMap().isSameAs(OperatorVector_[0]->getDomainMap())\n";
#else
            if (this->Verbose_) std::cerr << "SumOperator<SC,NO>::addOperator(SchwarzOperatorPtr op)\t\t!op->getDomainMap().isSameAs(OperatorVector_[0]->getDomainMap())\n";
#endif
            ret -= 1;
        }
        if (!op->getRangeMap().isSameAs(OperatorVector_[0]->getRangeMap())){
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            if (this->Verbose_) std::cerr << "SumOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr op)\t\t!op->getRangeMap().isSameAs(OperatorVector_[0]->getRangeMap())\n";
#else
            if (this->Verbose_) std::cerr << "SumOperator<SC,NO>::addOperator(SchwarzOperatorPtr op)\t\t!op->getRangeMap().isSameAs(OperatorVector_[0]->getRangeMap())\n";
#endif
            ret -= 10;
        }
        OperatorVector_[iD] = op;
        return ret;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int SumOperator<SC,LO,GO,NO>::enableOperator(UN iD,
#else
    template <class SC,class NO>
    int SumOperator<SC,NO>::enableOperator(UN iD,
#endif
                                                 bool enable)
  {
      FROSCH_TIMER_START_LEVELID(enableOperatorTime,"SumOperator::enableOperatorOperator");
      EnableOperators_[iD] = enable;
      return 0;
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    typename SumOperator<SC,LO,GO,NO>::UN SumOperator<SC,LO,GO,NO>::getNumOperators()
#else
    template <class SC,class NO>
    typename SumOperator<SC,NO>::UN SumOperator<SC,NO>::getNumOperators()
#endif
    {
      return OperatorVector_.size();
    }
}

#endif
