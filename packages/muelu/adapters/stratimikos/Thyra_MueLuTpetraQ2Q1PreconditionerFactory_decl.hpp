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
#ifndef THYRA_MUELU_TPETRA_Q2Q1PRECONDITIONER_FACTORY_DECL_HPP
#define THYRA_MUELU_TPETRA_Q2Q1PRECONDITIONER_FACTORY_DECL_HPP
#ifdef HAVE_MUELU_EXPERIMENTAL


#include "Thyra_PreconditionerFactoryBase.hpp"

#include "Kokkos_DefaultNode.hpp"

#include <Teko_Utilities.hpp>
#include <Xpetra_Matrix_fwd.hpp>

#include "MueLu_FactoryBase.hpp"

namespace Thyra {

  /** \brief Concrete preconditioner factory subclass based on MueLu.
   *
   * ToDo: Finish documentation!
   */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node = KokkosClassic::DefaultNode::DefaultNodeType>
#else
  template <class Scalar, class Node = KokkosClassic::DefaultNode::DefaultNodeType>
#endif
  class MueLuTpetraQ2Q1PreconditionerFactory : public PreconditionerFactoryBase<Scalar> {
  private:
#ifndef TPETRA_ENABLE_TEMPLATE_ORDINALS
    using LocalOrdinal = typename Tpetra::Map<>::local_ordinal_type;
    using GlobalOrdinal = typename Tpetra::Map<>::global_ordinal_type;
#endif
    typedef Scalar          SC;
    typedef LocalOrdinal    LO;
    typedef GlobalOrdinal   GO;
    typedef Node            NO;

  public:

    /** @name Constructors/initializers/accessors */
    //@{

    /** \brief . */
    MueLuTpetraQ2Q1PreconditionerFactory();
    //@}

    /** @name Overridden from PreconditionerFactoryBase */
    //@{

    /** \brief . */
    bool isCompatible( const LinearOpSourceBase<SC> &fwdOp ) const;
    /** \brief . */
    Teuchos::RCP<PreconditionerBase<SC> > createPrec() const;
    /** \brief . */
    void initializePrec(const Teuchos::RCP<const LinearOpSourceBase<SC> > &fwdOp, PreconditionerBase<SC> *prec, const ESupportSolveUse supportSolveUse) const;
    /** \brief . */
    void uninitializePrec(PreconditionerBase<SC> *prec, Teuchos::RCP<const LinearOpSourceBase<SC> > *fwdOp, ESupportSolveUse *supportSolveUse) const;
    //@}

    /** @name Overridden from Teuchos::ParameterListAcceptor */
    //@{

    /** \brief . */
    void                                          setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& paramList);
    /** \brief . */
    Teuchos::RCP<Teuchos::ParameterList>          unsetParameterList();
    /** \brief . */
    Teuchos::RCP<Teuchos::ParameterList>          getNonconstParameterList();
    /** \brief . */
    Teuchos::RCP<const Teuchos::ParameterList>    getParameterList() const;
    /** \brief . */
    Teuchos::RCP<const Teuchos::ParameterList>    getValidParameters() const;
    //@}

    /** \name Public functions overridden from Describable. */
    //@{

    /** \brief . */
    std::string description() const;

    // ToDo: Add an override of describe(...) to give more detail!

    //@}

  private:

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP<MueLu::TpetraOperator<SC,LO,GO,NO> >
#else
    Teuchos::RCP<MueLu::TpetraOperator<SC,NO> >
#endif
    Q2Q1MkPrecond(const ParameterList& paramList,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                  const Teuchos::RCP<Tpetra::MultiVector<SC,LO,GO,NO> >& velCoords,
                  const Teuchos::RCP<Tpetra::MultiVector<SC,LO,GO,NO> >& presCoords,
#else
                  const Teuchos::RCP<Tpetra::MultiVector<SC,NO> >& velCoords,
                  const Teuchos::RCP<Tpetra::MultiVector<SC,NO> >& presCoords,
#endif
                  const Teuchos::ArrayRCP<LO>& p2vMap,
                  const Teko::LinearOp& thA11, const Teko::LinearOp& thA12, const Teko::LinearOp& thA21, const Teko::LinearOp& thA11_9Pt) const;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP<Xpetra::Matrix<SC,LO,GO,NO> > Absolute    (const Xpetra::Matrix<SC,LO,GO,NO>& A) const;
    Teuchos::RCP<Xpetra::Matrix<SC,LO,GO,NO> > FilterMatrix(Xpetra::Matrix<SC,LO,GO,NO>& A, Xpetra::Matrix<SC,LO,GO,NO>& Pattern, SC dropTol) const;
#else
    Teuchos::RCP<Xpetra::Matrix<SC,NO> > Absolute    (const Xpetra::Matrix<SC,NO>& A) const;
    Teuchos::RCP<Xpetra::Matrix<SC,NO> > FilterMatrix(Xpetra::Matrix<SC,NO>& A, Xpetra::Matrix<SC,NO>& Pattern, SC dropTol) const;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void SetDependencyTree     (MueLu::FactoryManager<SC,LO,GO,NO>& M, const ParameterList& paramList) const;
    void SetBlockDependencyTree(MueLu::FactoryManager<SC,LO,GO,NO>& M, LO row, LO col, const std::string& mode, const ParameterList& paramList)  const;
#else
    void SetDependencyTree     (MueLu::FactoryManager<SC,NO>& M, const ParameterList& paramList) const;
    void SetBlockDependencyTree(MueLu::FactoryManager<SC,NO>& M, LO row, LO col, const std::string& mode, const ParameterList& paramList)  const;
#endif

    RCP<MueLu::FactoryBase> GetSmoother(const std::string& type, const ParameterList& paramList, bool coarseSolver) const;

    Teuchos::RCP<Teuchos::ParameterList> paramList_;

  };

} // namespace Thyra
#endif
#endif // THYRA_MUELU_TPETRA_Q2Q1PRECONDITIONER_FACTORY_DECL_HPP
