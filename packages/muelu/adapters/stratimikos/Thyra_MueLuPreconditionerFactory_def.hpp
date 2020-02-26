// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
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
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef THYRA_MUELU_PRECONDITIONER_FACTORY_DEF_HPP
#define THYRA_MUELU_PRECONDITIONER_FACTORY_DEF_HPP

#include "Thyra_MueLuPreconditionerFactory_decl.hpp"

#ifdef HAVE_MUELU_STRATIMIKOS

namespace Thyra {

  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;


  // Constructors/initializers/accessors

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  MueLuPreconditionerFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::MueLuPreconditionerFactory() :
#else
  template <class Scalar, class Node>
  MueLuPreconditionerFactory<Scalar,Node>::MueLuPreconditionerFactory() :
#endif
      paramList_(rcp(new ParameterList()))
  {}

  // Overridden from PreconditionerFactoryBase

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  bool MueLuPreconditionerFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::isCompatible(const LinearOpSourceBase<Scalar>& fwdOpSrc) const {
#else
  template <class Scalar, class Node>
  bool MueLuPreconditionerFactory<Scalar,Node>::isCompatible(const LinearOpSourceBase<Scalar>& fwdOpSrc) const {
#endif
    const RCP<const LinearOpBase<Scalar> > fwdOp = fwdOpSrc.getOp();

#ifdef HAVE_MUELU_TPETRA
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    if (Xpetra::ThyraUtils<Scalar,LocalOrdinal,GlobalOrdinal,Node>::isTpetra(fwdOp)) return true;
#else
    if (Xpetra::ThyraUtils<Scalar,Node>::isTpetra(fwdOp)) return true;
#endif
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    if (Xpetra::ThyraUtils<Scalar,LocalOrdinal,GlobalOrdinal,Node>::isBlockedOperator(fwdOp)) return true;
#else
    if (Xpetra::ThyraUtils<Scalar,Node>::isBlockedOperator(fwdOp)) return true;
#endif

    return false;
  }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<PreconditionerBase<Scalar> > MueLuPreconditionerFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::createPrec() const {
#else
  template <class Scalar, class Node>
  RCP<PreconditionerBase<Scalar> > MueLuPreconditionerFactory<Scalar,Node>::createPrec() const {
#endif
    return Teuchos::rcp(new DefaultPreconditioner<Scalar>);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void MueLuPreconditionerFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  template <class Scalar, class Node>
  void MueLuPreconditionerFactory<Scalar,Node>::
#endif
  initializePrec(const RCP<const LinearOpSourceBase<Scalar> >& fwdOpSrc, PreconditionerBase<Scalar>* prec, const ESupportSolveUse supportSolveUse) const {
    using Teuchos::rcp_dynamic_cast;

    // we are using typedefs here, since we are using objects from different packages (Xpetra, Thyra,...)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>                     XpMap;
    typedef Xpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>      XpOp;
    typedef Xpetra::ThyraUtils<Scalar,LocalOrdinal,GlobalOrdinal,Node>       XpThyUtils;
    typedef Xpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>        XpCrsMat;
    typedef Xpetra::BlockedCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> XpBlockedCrsMat;
    typedef Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>           XpMat;
    typedef Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>      XpMultVec;
    typedef Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::coordinateType,LocalOrdinal,GlobalOrdinal,Node>      XpMultVecDouble;
#else
    typedef Xpetra::Map<Node>                     XpMap;
    typedef Xpetra::Operator<Scalar, Node>      XpOp;
    typedef Xpetra::ThyraUtils<Scalar,Node>       XpThyUtils;
    typedef Xpetra::CrsMatrix<Scalar,Node>        XpCrsMat;
    typedef Xpetra::BlockedCrsMatrix<Scalar,Node> XpBlockedCrsMat;
    typedef Xpetra::Matrix<Scalar,Node>           XpMat;
    typedef Xpetra::MultiVector<Scalar,Node>      XpMultVec;
    typedef Xpetra::MultiVector<typename Teuchos::ScalarTraits<Scalar>::coordinateType,Node>      XpMultVecDouble;
#endif
    typedef Thyra::LinearOpBase<Scalar>                                      ThyLinOpBase;
#ifdef HAVE_MUELU_TPETRA
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef MueLu::TpetraOperator<Scalar,LocalOrdinal,GlobalOrdinal,Node> MueTpOp;
    typedef Tpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node>      TpOp;
    typedef Thyra::TpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node> ThyTpLinOp;
#else
    typedef MueLu::TpetraOperator<Scalar,Node> MueTpOp;
    typedef Tpetra::Operator<Scalar,Node>      TpOp;
    typedef Thyra::TpetraLinearOp<Scalar,Node> ThyTpLinOp;
#endif
#endif

    // Check precondition
    TEUCHOS_ASSERT(Teuchos::nonnull(fwdOpSrc));
    TEUCHOS_ASSERT(this->isCompatible(*fwdOpSrc));
    TEUCHOS_ASSERT(prec);

    // Create a copy, as we may remove some things from the list
    ParameterList paramList = *paramList_;

    // Retrieve wrapped concrete Xpetra matrix from FwdOp
    const RCP<const ThyLinOpBase> fwdOp = fwdOpSrc->getOp();
    TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(fwdOp));

    // Check whether it is Epetra/Tpetra
    bool bIsEpetra  = XpThyUtils::isEpetra(fwdOp);
    bool bIsTpetra  = XpThyUtils::isTpetra(fwdOp);
    bool bIsBlocked = XpThyUtils::isBlockedOperator(fwdOp);
    TEUCHOS_TEST_FOR_EXCEPT((bIsEpetra == true  && bIsTpetra == true));
    TEUCHOS_TEST_FOR_EXCEPT((bIsEpetra == bIsTpetra) && bIsBlocked == false);
    TEUCHOS_TEST_FOR_EXCEPT((bIsEpetra != bIsTpetra) && bIsBlocked == true);

    RCP<XpMat> A = Teuchos::null;
    if(bIsBlocked) {
      Teuchos::RCP<const Thyra::BlockedLinearOpBase<Scalar> > ThyBlockedOp =
          Teuchos::rcp_dynamic_cast<const Thyra::BlockedLinearOpBase<Scalar> >(fwdOp);
      TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(ThyBlockedOp));

      TEUCHOS_TEST_FOR_EXCEPT(ThyBlockedOp->blockExists(0,0)==false);

      Teuchos::RCP<const LinearOpBase<Scalar> > b00 = ThyBlockedOp->getBlock(0,0);
      TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(b00));

      RCP<const XpCrsMat > xpetraFwdCrsMat00 = XpThyUtils::toXpetra(b00);
      TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(xpetraFwdCrsMat00));

      // MueLu needs a non-const object as input
      RCP<XpCrsMat> xpetraFwdCrsMatNonConst00 = Teuchos::rcp_const_cast<XpCrsMat>(xpetraFwdCrsMat00);
      TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(xpetraFwdCrsMatNonConst00));

      // wrap the forward operator as an Xpetra::Matrix that MueLu can work with
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<XpMat> A00 = rcp(new Xpetra::CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>(xpetraFwdCrsMatNonConst00));
#else
      RCP<XpMat> A00 = rcp(new Xpetra::CrsMatrixWrap<Scalar,Node>(xpetraFwdCrsMatNonConst00));
#endif
      TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(A00));

      RCP<const XpMap> rowmap00 = A00->getRowMap();
      RCP< const Teuchos::Comm< int > > comm = rowmap00->getComm();

      // create a Xpetra::BlockedCrsMatrix which derives from Xpetra::Matrix that MueLu can work with
      RCP<XpBlockedCrsMat> bMat = Teuchos::rcp(new XpBlockedCrsMat(ThyBlockedOp, comm));
      TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(bMat));

      // save blocked matrix
      A = bMat;
    } else {
      RCP<const XpCrsMat > xpetraFwdCrsMat = XpThyUtils::toXpetra(fwdOp);
      TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(xpetraFwdCrsMat));

      // MueLu needs a non-const object as input
      RCP<XpCrsMat> xpetraFwdCrsMatNonConst = Teuchos::rcp_const_cast<XpCrsMat>(xpetraFwdCrsMat);
      TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(xpetraFwdCrsMatNonConst));

      // wrap the forward operator as an Xpetra::Matrix that MueLu can work with
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      A = rcp(new Xpetra::CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>(xpetraFwdCrsMatNonConst));
#else
      A = rcp(new Xpetra::CrsMatrixWrap<Scalar,Node>(xpetraFwdCrsMatNonConst));
#endif
    }
    TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(A));

    // Retrieve concrete preconditioner object
    const Teuchos::Ptr<DefaultPreconditioner<Scalar> > defaultPrec = Teuchos::ptr(dynamic_cast<DefaultPreconditioner<Scalar> *>(prec));
    TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(defaultPrec));

    // extract preconditioner operator
    RCP<ThyLinOpBase> thyra_precOp = Teuchos::null;
    thyra_precOp = rcp_dynamic_cast<Thyra::LinearOpBase<Scalar> >(defaultPrec->getNonconstUnspecifiedPrecOp(), true);

    // Variable for multigrid hierarchy: either build a new one or reuse the existing hierarchy
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<MueLu::Hierarchy<Scalar,LocalOrdinal,GlobalOrdinal,Node> > H = Teuchos::null;
#else
    RCP<MueLu::Hierarchy<Scalar,Node> > H = Teuchos::null;
#endif

    // make a decision whether to (re)build the multigrid preconditioner or reuse the old one
    // rebuild preconditioner if startingOver == true
    // reuse preconditioner if startingOver == false
    const bool startingOver = (thyra_precOp.is_null() || !paramList.isParameter("reuse: type") || paramList.get<std::string>("reuse: type") == "none");

    if (startingOver == true) {
      // extract coordinates from parameter list
      RCP<XpMultVecDouble> coordinates = Teuchos::null;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      coordinates = MueLu::Utilities<Scalar,LocalOrdinal,GlobalOrdinal,Node>::ExtractCoordinatesFromParameterList(paramList);
#else
      coordinates = MueLu::Utilities<Scalar,Node>::ExtractCoordinatesFromParameterList(paramList);
#endif

      // TODO check for Xpetra or Thyra vectors?
      RCP<XpMultVec> nullspace = Teuchos::null;
#ifdef HAVE_MUELU_TPETRA
      if (bIsTpetra) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        typedef Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> tMV;
#else
        typedef Tpetra::MultiVector<Scalar, Node> tMV;
#endif
        RCP<tMV> tpetra_nullspace = Teuchos::null;
        if (paramList.isType<Teuchos::RCP<tMV> >("Nullspace")) {
          tpetra_nullspace = paramList.get<RCP<tMV> >("Nullspace");
          paramList.remove("Nullspace");
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
          nullspace = MueLu::TpetraMultiVector_To_XpetraMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>(tpetra_nullspace);
#else
          nullspace = MueLu::TpetraMultiVector_To_XpetraMultiVector<Scalar,Node>(tpetra_nullspace);
#endif
          TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(nullspace));
        }
      }
#endif
      // build a new MueLu hierarchy
      ParameterList& userParamList = paramList.sublist("user data");
      if(Teuchos::nonnull(coordinates)) {
        userParamList.set<RCP<XpMultVecDouble> >("Coordinates", coordinates);
      }
      if(Teuchos::nonnull(nullspace)) {
        userParamList.set<RCP<XpMultVec> >("Nullspace", nullspace);
      }
      H = MueLu::CreateXpetraPreconditioner(A, paramList);

    } else {
      // reuse old MueLu hierarchy stored in MueLu Tpetra/Epetra operator and put in new matrix

      // get old MueLu hierarchy
#if defined(HAVE_MUELU_TPETRA)
      if (bIsTpetra) {

        RCP<ThyTpLinOp> tpetr_precOp = rcp_dynamic_cast<ThyTpLinOp>(thyra_precOp);
        RCP<MueTpOp>    muelu_precOp = rcp_dynamic_cast<MueTpOp>(tpetr_precOp->getTpetraOperator(),true);

        H = muelu_precOp->GetHierarchy();
      }
#endif
      // TODO add the blocked matrix case here...

      TEUCHOS_TEST_FOR_EXCEPTION(!H->GetNumLevels(), MueLu::Exceptions::RuntimeError,
                                 "Thyra::MueLuPreconditionerFactory: Hierarchy has no levels in it");
      TEUCHOS_TEST_FOR_EXCEPTION(!H->GetLevel(0)->IsAvailable("A"), MueLu::Exceptions::RuntimeError,
                                 "Thyra::MueLuPreconditionerFactory: Hierarchy has no fine level operator");
      RCP<MueLu::Level> level0 = H->GetLevel(0);
      RCP<XpOp>    O0 = level0->Get<RCP<XpOp> >("A");
      RCP<XpMat>   A0 = rcp_dynamic_cast<XpMat>(O0);

      if (!A0.is_null()) {
        // If a user provided a "number of equations" argument in a parameter list
        // during the initial setup, we must honor that settings and reuse it for
        // all consequent setups.
        A->SetFixedBlockSize(A0->GetFixedBlockSize());
      }

      // set new matrix
      level0->Set("A", A);

      H->SetupRe();
    }

    // wrap hierarchy H in thyraPrecOp
    RCP<ThyLinOpBase > thyraPrecOp = Teuchos::null;
#if defined(HAVE_MUELU_TPETRA)
    if (bIsTpetra) {
      RCP<MueTpOp> muelu_tpetraOp = rcp(new MueTpOp(H));
      TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(muelu_tpetraOp));
      RCP<TpOp> tpOp = Teuchos::rcp_dynamic_cast<TpOp>(muelu_tpetraOp);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      thyraPrecOp = Thyra::createLinearOp<Scalar, LocalOrdinal, GlobalOrdinal, Node>(tpOp);
#else
      thyraPrecOp = Thyra::createLinearOp<Scalar, Node>(tpOp);
#endif
    }
#endif

    if(bIsBlocked) {
      TEUCHOS_TEST_FOR_EXCEPT(Teuchos::nonnull(thyraPrecOp));

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      typedef MueLu::XpetraOperator<Scalar,LocalOrdinal,GlobalOrdinal,Node>    MueXpOp;
#else
      typedef MueLu::XpetraOperator<Scalar,Node>    MueXpOp;
#endif
      //typedef Thyra::XpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node>    ThyXpLinOp; // unused
      const RCP<MueXpOp> muelu_xpetraOp = rcp(new MueXpOp(H));

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<const VectorSpaceBase<Scalar> > thyraRangeSpace  = Xpetra::ThyraUtils<Scalar,LocalOrdinal,GlobalOrdinal,Node>::toThyra(muelu_xpetraOp->getRangeMap());
      RCP<const VectorSpaceBase<Scalar> > thyraDomainSpace = Xpetra::ThyraUtils<Scalar,LocalOrdinal,GlobalOrdinal,Node>::toThyra(muelu_xpetraOp->getDomainMap());
#else
      RCP<const VectorSpaceBase<Scalar> > thyraRangeSpace  = Xpetra::ThyraUtils<Scalar,Node>::toThyra(muelu_xpetraOp->getRangeMap());
      RCP<const VectorSpaceBase<Scalar> > thyraDomainSpace = Xpetra::ThyraUtils<Scalar,Node>::toThyra(muelu_xpetraOp->getDomainMap());
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP <Xpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> > xpOp = Teuchos::rcp_dynamic_cast<Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(muelu_xpetraOp);
      thyraPrecOp = Thyra::xpetraLinearOp<Scalar, LocalOrdinal, GlobalOrdinal, Node>(thyraRangeSpace, thyraDomainSpace,xpOp);
#else
      RCP <Xpetra::Operator<Scalar, Node> > xpOp = Teuchos::rcp_dynamic_cast<Xpetra::Operator<Scalar,Node> >(muelu_xpetraOp);
      thyraPrecOp = Thyra::xpetraLinearOp<Scalar, Node>(thyraRangeSpace, thyraDomainSpace,xpOp);
#endif
    }

    TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(thyraPrecOp));

    defaultPrec->initializeUnspecified(thyraPrecOp);

  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void MueLuPreconditionerFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
#else
  template <class Scalar, class Node>
  void MueLuPreconditionerFactory<Scalar,Node>::
#endif
  uninitializePrec(PreconditionerBase<Scalar>* prec, RCP<const LinearOpSourceBase<Scalar> >* fwdOp, ESupportSolveUse* supportSolveUse) const {
    TEUCHOS_ASSERT(prec);

    // Retrieve concrete preconditioner object
    const Teuchos::Ptr<DefaultPreconditioner<Scalar> > defaultPrec = Teuchos::ptr(dynamic_cast<DefaultPreconditioner<Scalar> *>(prec));
    TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(defaultPrec));

    if (fwdOp) {
      // TODO: Implement properly instead of returning default value
      *fwdOp = Teuchos::null;
    }

    if (supportSolveUse) {
      // TODO: Implement properly instead of returning default value
      *supportSolveUse = Thyra::SUPPORT_SOLVE_UNSPECIFIED;
    }

    defaultPrec->uninitialize();
  }


  // Overridden from ParameterListAcceptor
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void MueLuPreconditionerFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::setParameterList(RCP<ParameterList> const& paramList) {
#else
  template <class Scalar, class Node>
  void MueLuPreconditionerFactory<Scalar,Node>::setParameterList(RCP<ParameterList> const& paramList) {
#endif
    TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(paramList));
    paramList_ = paramList;
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<ParameterList> MueLuPreconditionerFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getNonconstParameterList() {
#else
  template <class Scalar, class Node>
  RCP<ParameterList> MueLuPreconditionerFactory<Scalar,Node>::getNonconstParameterList() {
#endif
    return paramList_;
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<ParameterList> MueLuPreconditionerFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::unsetParameterList() {
#else
  template <class Scalar, class Node>
  RCP<ParameterList> MueLuPreconditionerFactory<Scalar,Node>::unsetParameterList() {
#endif
    RCP<ParameterList> savedParamList = paramList_;
    paramList_ = Teuchos::null;
    return savedParamList;
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const ParameterList> MueLuPreconditionerFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getParameterList() const {
#else
  template <class Scalar, class Node>
  RCP<const ParameterList> MueLuPreconditionerFactory<Scalar,Node>::getParameterList() const {
#endif
    return paramList_;
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const ParameterList> MueLuPreconditionerFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getValidParameters() const {
#else
  template <class Scalar, class Node>
  RCP<const ParameterList> MueLuPreconditionerFactory<Scalar,Node>::getValidParameters() const {
#endif
    static RCP<const ParameterList> validPL;

    if (Teuchos::is_null(validPL))
      validPL = rcp(new ParameterList());

    return validPL;
  }

  // Public functions overridden from Teuchos::Describable
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  std::string MueLuPreconditionerFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::description() const {
#else
  template <class Scalar, class Node>
  std::string MueLuPreconditionerFactory<Scalar,Node>::description() const {
#endif
    return "Thyra::MueLuPreconditionerFactory";
  }
} // namespace Thyra

#endif // HAVE_MUELU_STRATIMIKOS

#endif // ifdef THYRA_MUELU_PRECONDITIONER_FACTORY_DEF_HPP
