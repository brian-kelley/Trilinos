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

#ifndef THYRA_FROSCH_LINEAR_OP_DEF_HPP
#define THYRA_FROSCH_LINEAR_OP_DEF_HPP

#include "Thyra_FROSchLinearOp_decl.hpp"


#ifdef HAVE_SHYLU_DDFROSCH_THYRA
namespace Thyra {

    using namespace std;
    using namespace FROSch;
    using namespace Teuchos;
    using namespace Xpetra;

    // Constructors/initializers
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    FROSchLinearOp<SC,LO,GO,NO>::FROSchLinearOp()
#else
    template <class SC, class NO>
    FROSchLinearOp<SC,NO>::FROSchLinearOp()
#endif
    {
        
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    void FROSchLinearOp<SC,LO,GO,NO>::initialize(const RCP<const VectorSpaceBase<SC> > &rangeSpace,
#else
    template <class SC, class NO>
    void FROSchLinearOp<SC,NO>::initialize(const RCP<const VectorSpaceBase<SC> > &rangeSpace,
#endif
                                                 const RCP<const VectorSpaceBase<SC> > &domainSpace,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                                                 const RCP<Operator<SC,LO,GO,NO> > &xpetraOperator,
#else
                                                 const RCP<Operator<SC,NO> > &xpetraOperator,
#endif
                                                 bool bIsEpetra,
                                                 bool bIsTpetra)
    {
        initializeImpl(rangeSpace, domainSpace, xpetraOperator,bIsEpetra,bIsTpetra);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    void FROSchLinearOp<SC,LO,GO,NO>::constInitialize(const RCP<const VectorSpaceBase<SC> > &rangeSpace,
#else
    template <class SC, class NO>
    void FROSchLinearOp<SC,NO>::constInitialize(const RCP<const VectorSpaceBase<SC> > &rangeSpace,
#endif
                                                      const RCP<const VectorSpaceBase<SC> > &domainSpace,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                                                      const RCP<const Operator<SC,LO,GO,NO> > &xpetraOperator,
#else
                                                      const RCP<const Operator<SC,NO> > &xpetraOperator,
#endif
                                                      bool bIsEpetra,
                                                      bool bIsTpetra)
    {
        initializeImpl(rangeSpace, domainSpace, xpetraOperator,bIsEpetra,bIsTpetra);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    RCP<Operator<SC,LO,GO,NO> > FROSchLinearOp<SC,LO,GO,NO>::getXpetraOperator()
#else
    template <class SC, class NO>
    RCP<Operator<SC,NO> > FROSchLinearOp<SC,NO>::getXpetraOperator()
#endif
    {
        return xpetraOperator_.getNonconstObj();
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    RCP<const Operator<SC,LO,GO,NO> > FROSchLinearOp<SC,LO,GO,NO>::getConstXpetraOperator() const
#else
    template <class SC, class NO>
    RCP<const Operator<SC,NO> > FROSchLinearOp<SC,NO>::getConstXpetraOperator() const
#endif
    {
        return xpetraOperator_;
    }

    // Public Overridden functions from LinearOpBase

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    RCP<const VectorSpaceBase<SC> > FROSchLinearOp<SC,LO,GO,NO>::range() const
#else
    template <class SC, class NO>
    RCP<const VectorSpaceBase<SC> > FROSchLinearOp<SC,NO>::range() const
#endif
    {
        return rangeSpace_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    RCP<const VectorSpaceBase<SC> > FROSchLinearOp<SC,LO,GO,NO>::domain() const
#else
    template <class SC, class NO>
    RCP<const VectorSpaceBase<SC> > FROSchLinearOp<SC,NO>::domain() const
#endif
    {
        return domainSpace_;
    }

    // Protected Overridden functions from LinearOpBase

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    bool FROSchLinearOp<SC,LO,GO,NO>::opSupportedImpl(EOpTransp M_trans) const
#else
    template <class SC, class NO>
    bool FROSchLinearOp<SC,NO>::opSupportedImpl(EOpTransp M_trans) const
#endif
    {
        if (is_null(xpetraOperator_))
        return false;

        if (M_trans == NOTRANS)
        return true;

        if (M_trans == CONJ) {
            // For non-complex scalars, CONJ is always supported since it is equivalent to NO_TRANS.
            // For complex scalars, Xpetra does not support conjugation without transposition.
            return !ScalarTraits<SC>::isComplex;
        }

        return xpetraOperator_->hasTransposeApply();
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
    void FROSchLinearOp<SC,LO,GO,NO>::applyImpl(const EOpTransp M_trans,
#else
    template <class SC, class NO>
    void FROSchLinearOp<SC,NO>::applyImpl(const EOpTransp M_trans,
#endif
                                                const MultiVectorBase<SC> &X_in,
                                                const Ptr<MultiVectorBase<SC> > &Y_inout,
                                                const SC alpha,
                                                const SC beta) const
    {
        FROSCH_ASSERT(getConstXpetraOperator()!=null,"XpetraLinearOp::applyImpl: internal Operator is null.");
        RCP< const Comm<int> > comm = getConstXpetraOperator()->getRangeMap()->getComm();
        //Transform to Xpetra MultiVector
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        RCP<MultiVector<SC,LO,GO,NO> > xY;
#else
        RCP<MultiVector<SC,NO> > xY;
#endif

        ETransp transp = NO_TRANS;
        switch (M_trans) {
            case NOTRANS:   transp = NO_TRANS;          break;
            case TRANS:     transp = Teuchos::TRANS;    break;
            case CONJTRANS: transp = CONJ_TRANS;        break;
            default: FROSCH_ASSERT(false,"Thyra::XpetraLinearOp::apply. Unknown value for M_trans. Only NOTRANS, TRANS and CONJTRANS are supported.");
        }
        //Epetra NodeType
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
        const EOpTransp real_M_trans = real_trans(M_trans);

        if (this->bIsEpetra_) {
            const RCP<const VectorSpaceBase<double> > XY_domain = X_in.domain();

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            RCP<const Map<LO,GO,NO> > DomainM = this->xpetraOperator_->getDomainMap();
#else
            RCP<const Map<NO> > DomainM = this->xpetraOperator_->getDomainMap();
#endif
            RCP<const EpetraMapT<GO,NO> > eDomainM = rcp_dynamic_cast<const EpetraMapT<GO,NO> >(DomainM);
            const Epetra_Map epetraDomain = eDomainM->getEpetra_Map();

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            RCP<const Map<LO,GO,NO> > RangeM = this->xpetraOperator_->getRangeMap();
#else
            RCP<const Map<NO> > RangeM = this->xpetraOperator_->getRangeMap();
#endif
            RCP<const EpetraMapT<GO,NO> > eRangeM = rcp_dynamic_cast<const EpetraMapT<GO,NO> >(RangeM);
            const Epetra_Map epetraRange = eRangeM->getEpetra_Map();

            RCP<const Epetra_MultiVector> X;
            RCP<Epetra_MultiVector> Y;

            THYRA_FUNC_TIME_MONITOR_DIFF("Thyra::EpetraLinearOp::euclideanApply: Convert MultiVectors", MultiVectors);
            // X
            X = get_Epetra_MultiVector(real_M_trans==NOTRANS ? epetraDomain: epetraRange, X_in );
            RCP<Epetra_MultiVector> X_nonconst = rcp_const_cast<Epetra_MultiVector>(X);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            RCP<MultiVector<SC,LO,GO,NO> > xX = FROSch::ConvertToXpetra<SC,LO,GO,NO>::ConvertMultiVector(UseEpetra,*X_nonconst,comm);
#else
            RCP<MultiVector<SC,NO> > xX = FROSch::ConvertToXpetra<SC,NO>::ConvertMultiVector(UseEpetra,*X_nonconst,comm);
#endif
            // Y
            Y = get_Epetra_MultiVector(real_M_trans==NOTRANS ? epetraRange: epetraDomain, *Y_inout );
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            xY = FROSch::ConvertToXpetra<SC,LO,GO,NO>::ConvertMultiVector(UseEpetra,*Y,comm);
#else
            xY = FROSch::ConvertToXpetra<SC,NO>::ConvertMultiVector(UseEpetra,*Y,comm);
#endif
            xpetraOperator_->apply(*xX, *xY, transp, alpha, beta);

        } //Tpetra NodeType
        else
#endif
        if (bIsTpetra_) {
            // Convert input vector to Xpetra
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            const RCP<const Tpetra::MultiVector<SC,LO,GO,NO> > xTpMultVec = Thyra::TpetraOperatorVectorExtraction<SC,LO,GO,NO>::getConstTpetraMultiVector(rcpFromRef(X_in));
#else
            const RCP<const Tpetra::MultiVector<SC,NO> > xTpMultVec = Thyra::TpetraOperatorVectorExtraction<SC,NO>::getConstTpetraMultiVector(rcpFromRef(X_in));
#endif
            TEUCHOS_TEST_FOR_EXCEPT(is_null(xTpMultVec));
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            RCP<Tpetra::MultiVector<SC,LO,GO,NO> > tpNonConstMultVec = rcp_const_cast<Tpetra::MultiVector<SC,LO,GO,NO> >(xTpMultVec);
#else
            RCP<Tpetra::MultiVector<SC,NO> > tpNonConstMultVec = rcp_const_cast<Tpetra::MultiVector<SC,NO> >(xTpMultVec);
#endif
            TEUCHOS_TEST_FOR_EXCEPT(is_null(tpNonConstMultVec));
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            const RCP<const Xpetra::MultiVector<SC,LO,GO,NO> > xX = rcp(new Xpetra::TpetraMultiVector<SC,LO,GO,NO>(tpNonConstMultVec));
#else
            const RCP<const Xpetra::MultiVector<SC,NO> > xX = rcp(new Xpetra::TpetraMultiVector<SC,NO>(tpNonConstMultVec));
#endif
            TEUCHOS_TEST_FOR_EXCEPT(is_null(xX));

            // Convert output vector to Xpetra
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            const RCP<Tpetra::MultiVector<SC,LO,GO,NO> > yTpMultVec = Thyra::TpetraOperatorVectorExtraction<SC,LO,GO,NO>::getTpetraMultiVector(rcpFromPtr(Y_inout));
#else
            const RCP<Tpetra::MultiVector<SC,NO> > yTpMultVec = Thyra::TpetraOperatorVectorExtraction<SC,NO>::getTpetraMultiVector(rcpFromPtr(Y_inout));
#endif
            TEUCHOS_TEST_FOR_EXCEPT(is_null(yTpMultVec));
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            xY = rcp(new Xpetra::TpetraMultiVector<SC,LO,GO,NO>(yTpMultVec));
#else
            xY = rcp(new Xpetra::TpetraMultiVector<SC,NO>(yTpMultVec));
#endif
            TEUCHOS_TEST_FOR_EXCEPT(is_null(xY));

            // Apply operator
            xpetraOperator_->apply(*xX, *xY, transp, alpha, beta);

        } else {
            FROSCH_ASSERT(false,"There is a problem with the underlying lib in FROSchLinearOp.");
            //std::cout<<"Only Implemented for Epetra and Tpetra\n";
        }

#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
        if (this->bIsEpetra_) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            RCP<MultiVectorBase<SC> > thyraX = rcp_const_cast<MultiVectorBase<SC> >(ThyraUtils<SC,LO,GO,NO>::toThyraMultiVector(xY));
#else
            RCP<MultiVectorBase<SC> > thyraX = rcp_const_cast<MultiVectorBase<SC> >(ThyraUtils<SC,NO>::toThyraMultiVector(xY));
#endif
            
            using ThySpmdVecSpaceBase = SpmdVectorSpaceBase<SC> ;
            RCP<const ThySpmdVecSpaceBase> mpi_vs = rcp_dynamic_cast<const ThySpmdVecSpaceBase>(rcpFromPtr(Y_inout)->range());
            
            TEUCHOS_TEST_FOR_EXCEPTION(mpi_vs == null, std::logic_error, "Failed to cast Thyra::VectorSpaceBase to Thyra::SpmdVectorSpaceBase.");
            const LO localOffset = ( mpi_vs != null ? mpi_vs->localOffset() : 0 );
            const LO localSubDim = ( mpi_vs != null ? mpi_vs->localSubDim() : rcpFromPtr(Y_inout)->range()->dim() );
            
            RCP<DetachedMultiVectorView<SC> > thyData = rcp(new DetachedMultiVectorView<SC>(*rcpFromPtr(Y_inout),Range1D(localOffset,localOffset+localSubDim-1)));
            
            // AH 08/14/2019 TODO: Is this necessary??
            for( size_t j = 0; j <xY->getNumVectors(); ++j) {
                ArrayRCP< const SC > xpData = xY->getData(j); // access const data from Xpetra object
                // loop over all local rows
                for( LO i = 0; i < localSubDim; ++i) {
                    (*thyData)(i,j) = xpData[i];
                }
            }
        }
#endif
    }

    // private

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO, class GO, class NO>
#else
    template <class SC, class NO>
#endif
    template<class XpetraOperator_t>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void FROSchLinearOp<SC,LO,GO,NO>::initializeImpl(const RCP<const VectorSpaceBase<SC> > &rangeSpace,
#else
    void FROSchLinearOp<SC,NO>::initializeImpl(const RCP<const VectorSpaceBase<SC> > &rangeSpace,
#endif
                                                     const RCP<const VectorSpaceBase<SC> > &domainSpace,
                                                     const RCP<XpetraOperator_t> &xpetraOperator,
                                                     bool bIsEpetra,
                                                     bool bIsTpetra)
    {
#ifdef THYRA_DEBUG
        TEUCHOS_ASSERT(nonnull(rangeSpace));
        TEUCHOS_ASSERT(nonnull(domainSpace));
        TEUCHOS_ASSERT(nonnull(xpetraOperator));
#endif
        rangeSpace_ = rangeSpace;
        domainSpace_ = domainSpace;
        xpetraOperator_ = xpetraOperator;
        bIsEpetra_ = bIsEpetra;
        bIsTpetra_ = bIsTpetra;
    }

} // namespace Thyra

#endif

#endif  // THYRA_XPETRA_LINEAR_OP_HPP
