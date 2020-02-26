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

#ifndef _FROSCH_OVERLAPPINGOPERATOR_DEF_HPP
#define _FROSCH_OVERLAPPINGOPERATOR_DEF_HPP

#include <FROSch_OverlappingOperator_decl.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    OverlappingOperator<SC,LO,GO,NO>::OverlappingOperator(ConstXMatrixPtr k,
#else
    template <class SC,class NO>
    OverlappingOperator<SC,NO>::OverlappingOperator(ConstXMatrixPtr k,
#endif
                                                          ParameterListPtr parameterList) :
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    SchwarzOperator<SC,LO,GO,NO> (k,parameterList)
#else
    SchwarzOperator<SC,NO> (k,parameterList)
#endif
    {
        FROSCH_TIMER_START_LEVELID(overlappingOperatorTime,"OverlappingOperator::OverlappingOperator");
        if (!this->ParameterList_->get("Combine Values in Overlap","Restricted").compare("Averaging")) {
            Combine_ = Averaging;
        } else if (!this->ParameterList_->get("Combine Values in Overlap","Restricted").compare("Full")) {
            Combine_ = Full;
        } else if (!this->ParameterList_->get("Combine Values in Overlap","Restricted").compare("Restricted")) {
            Combine_ = Restricted;
        }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    OverlappingOperator<SC,LO,GO,NO>::~OverlappingOperator()
#else
    template <class SC,class NO>
    OverlappingOperator<SC,NO>::~OverlappingOperator()
#endif
    {
        SubdomainSolver_.reset();
    }

    // Y = alpha * A^mode * X + beta * Y
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    void OverlappingOperator<SC,LO,GO,NO>::apply(const XMultiVector &x,
#else
    template <class SC,class NO>
    void OverlappingOperator<SC,NO>::apply(const XMultiVector &x,
#endif
                                                 XMultiVector &y,
                                                 bool usePreconditionerOnly,
                                                 ETransp mode,
                                                 SC alpha,
                                                 SC beta) const
    {
        FROSCH_TIMER_START_LEVELID(applyTime,"OverlappingOperator::apply");
        FROSCH_ASSERT(this->IsComputed_,"FROSch::OverlappingOperator : ERROR: OverlappingOperator has to be computed before calling apply()");
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        if (XTmp_.is_null()) XTmp_ = MultiVectorFactory<SC,LO,GO,NO>::Build(x.getMap(),x.getNumVectors());
#else
        if (XTmp_.is_null()) XTmp_ = MultiVectorFactory<SC,NO>::Build(x.getMap(),x.getNumVectors());
#endif
        *XTmp_ = x;
        if (!usePreconditionerOnly && mode == NO_TRANS) {
            this->K_->apply(x,*XTmp_,mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
        }
        // AH 11/28/2018: For Epetra, XOverlap_ will only have a view to the values of XOverlapTmp_. Therefore, xOverlapTmp should not be deleted before XOverlap_ is used.
        if (YOverlap_.is_null()) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            YOverlap_ = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMatrix_->getDomainMap(),x.getNumVectors());
#else
            YOverlap_ = MultiVectorFactory<SC,NO>::Build(OverlappingMatrix_->getDomainMap(),x.getNumVectors());
#endif
        } else {
            YOverlap_->replaceMap(OverlappingMatrix_->getDomainMap());
        }
        // AH 11/28/2018: replaceMap does not update the GlobalNumRows. Therefore, we have to create a new MultiVector on the serial Communicator. In Epetra, we can prevent to copy the MultiVector.
        if (XTmp_->getMap()->lib() == UseEpetra) {
#ifdef HAVE_XPETRA_EPETRA
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            if (XOverlapTmp_.is_null()) XOverlapTmp_ = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,x.getNumVectors());
#else
            if (XOverlapTmp_.is_null()) XOverlapTmp_ = MultiVectorFactory<SC,NO>::Build(OverlappingMap_,x.getNumVectors());
#endif
            XOverlapTmp_->doImport(*XTmp_,*Scatter_,INSERT);
            const RCP<const EpetraMultiVectorT<GO,NO> > xEpetraMultiVectorXOverlapTmp = rcp_dynamic_cast<const EpetraMultiVectorT<GO,NO> >(XOverlapTmp_);
            RCP<Epetra_MultiVector> epetraMultiVectorXOverlapTmp = xEpetraMultiVectorXOverlapTmp->getEpetra_MultiVector();
            const RCP<const EpetraMapT<GO,NO> >& xEpetraMap = rcp_dynamic_cast<const EpetraMapT<GO,NO> >(OverlappingMatrix_->getRangeMap());
            Epetra_BlockMap epetraMap = xEpetraMap->getEpetra_BlockMap();
            double *A;
            int MyLDA;
            epetraMultiVectorXOverlapTmp->ExtractView(&A,&MyLDA);
            RCP<Epetra_MultiVector> epetraMultiVectorXOverlap(new Epetra_MultiVector(::View,epetraMap,A,MyLDA,x.getNumVectors()));
            XOverlap_ = RCP<EpetraMultiVectorT<GO,NO> >(new EpetraMultiVectorT<GO,NO>(epetraMultiVectorXOverlap));
#else
            FROSCH_ASSERT(false,"HAVE_XPETRA_EPETRA not defined.");
#endif
        } else {
            if (XOverlap_.is_null()) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                XOverlap_ = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,x.getNumVectors());
#else
                XOverlap_ = MultiVectorFactory<SC,NO>::Build(OverlappingMap_,x.getNumVectors());
#endif
            } else {
                XOverlap_->replaceMap(OverlappingMap_);
            }
            XOverlap_->doImport(*XTmp_,*Scatter_,INSERT);
            XOverlap_->replaceMap(OverlappingMatrix_->getRangeMap());
        }
        SubdomainSolver_->apply(*XOverlap_,*YOverlap_,mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
        YOverlap_->replaceMap(OverlappingMap_);

        XTmp_->putScalar(ScalarTraits<SC>::zero());
        ConstXMapPtr yMap = y.getMap();
        ConstXMapPtr yOverlapMap = YOverlap_->getMap();
        if (Combine_ == Restricted){
            GO globID = 0;
            LO localID = 0;
            for (UN i=0; i<y.getNumVectors(); i++) {
                ConstSCVecPtr yOverlapData_i = YOverlap_->getData(i);
                for (UN j=0; j<yMap->getNodeNumElements(); j++) {
                    globID = yMap->getGlobalElement(j);
                    localID = yOverlapMap->getLocalElement(globID);
                    XTmp_->getDataNonConst(i)[j] = yOverlapData_i[localID];
                }
            }
        } else {
            XTmp_->doExport(*YOverlap_,*Scatter_,ADD);
        }
        if (Combine_ == Averaging) {
            ConstSCVecPtr scaling = Multiplicity_->getData(0);
            for (UN j=0; j<XTmp_->getNumVectors(); j++) {
                SCVecPtr values = XTmp_->getDataNonConst(j);
                for (UN i=0; i<values.size(); i++) {
                    values[i] = values[i] / scaling[i];
                }
            }
        }

        if (!usePreconditionerOnly && mode != NO_TRANS) {
            this->K_->apply(*XTmp_,*XTmp_,mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
        }
        y.update(alpha,*XTmp_,beta);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int OverlappingOperator<SC,LO,GO,NO>::initializeOverlappingOperator()
#else
    template <class SC,class NO>
    int OverlappingOperator<SC,NO>::initializeOverlappingOperator()
#endif
    {
        FROSCH_TIMER_START_LEVELID(initializeOverlappingOperatorTime,"OverlappingOperator::initializeOverlappingOperator");
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        Scatter_ = ImportFactory<LO,GO,NO>::Build(this->getDomainMap(),OverlappingMap_);
#else
        Scatter_ = ImportFactory<NO>::Build(this->getDomainMap(),OverlappingMap_);
#endif
        if (Combine_ == Averaging) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            Multiplicity_ = MultiVectorFactory<SC,LO,GO,NO>::Build(this->getRangeMap(),1);
#else
            Multiplicity_ = MultiVectorFactory<SC,NO>::Build(this->getRangeMap(),1);
#endif
            XMultiVectorPtr multiplicityRepeated;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            multiplicityRepeated = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,1);
#else
            multiplicityRepeated = MultiVectorFactory<SC,NO>::Build(OverlappingMap_,1);
#endif
            multiplicityRepeated->putScalar(ScalarTraits<SC>::one());
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            XExportPtr multiplicityExporter = ExportFactory<LO,GO,NO>::Build(multiplicityRepeated->getMap(),this->getRangeMap());
#else
            XExportPtr multiplicityExporter = ExportFactory<NO>::Build(multiplicityRepeated->getMap(),this->getRangeMap());
#endif
            Multiplicity_->doExport(*multiplicityRepeated,*multiplicityExporter,ADD);
        }

        return 0; // RETURN VALUE
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int OverlappingOperator<SC,LO,GO,NO>::computeOverlappingOperator()
#else
    template <class SC,class NO>
    int OverlappingOperator<SC,NO>::computeOverlappingOperator()
#endif
    {
        FROSCH_TIMER_START_LEVELID(computeOverlappingOperatorTime,"OverlappingOperator::computeOverlappingOperator");
        
        updateLocalOverlappingMatrices();

        bool reuseSymbolicFactorization = this->ParameterList_->get("Reuse: Symbolic Factorization",true);
        if (!this->IsComputed_) {
            reuseSymbolicFactorization = false;
        }

        if (!reuseSymbolicFactorization) {
            if (this->IsComputed_ && this->Verbose_) std::cout << "FROSch::OverlappingOperator : Recomputing the Symbolic Factorization" << std::endl;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            SubdomainSolver_.reset(new SubdomainSolver<SC,LO,GO,NO>(OverlappingMatrix_,sublist(this->ParameterList_,"Solver")));
#else
            SubdomainSolver_.reset(new SubdomainSolver<SC,NO>(OverlappingMatrix_,sublist(this->ParameterList_,"Solver")));
#endif
            SubdomainSolver_->initialize();
        } else {
            FROSCH_ASSERT(!SubdomainSolver_.is_null(),"FROSch::OverlappingOperator : ERROR: SubdomainSolver_.is_null()");
            SubdomainSolver_->resetMatrix(OverlappingMatrix_,true);
        }
        this->IsComputed_ = true;
        return SubdomainSolver_->compute();
    }
}

#endif
