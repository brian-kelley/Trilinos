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
#ifndef MUELU_XPETRAOPERATOR_DECL_HPP
#define MUELU_XPETRAOPERATOR_DECL_HPP

#include "MueLu_ConfigDefs.hpp"

#include <Xpetra_Operator.hpp>
#include <Xpetra_MultiVector.hpp>
#include "MueLu_Level.hpp"
#include "MueLu_Hierarchy_decl.hpp"

namespace MueLu {

/*!  @brief Wraps an existing MueLu::Hierarchy as a Xpetra::Operator.
*/
  template <class Scalar = DefaultScalar,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            class LocalOrdinal = DefaultLocalOrdinal,
            class GlobalOrdinal = DefaultGlobalOrdinal,
#endif
            class Node = DefaultNode>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  class XpetraOperator : public Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node> {
#else
  class XpetraOperator : public Xpetra::Operator<Scalar,Node> {
#endif
  protected:
#ifndef TPETRA_ENABLE_TEMPLATE_ORDINALS
    using LocalOrdinal = typename Tpetra::Map<>::local_ordinal_type;
    using GlobalOrdinal = typename Tpetra::Map<>::global_ordinal_type;
#endif
    XpetraOperator() { }
  public:

    //! @name Constructor/Destructor
    //@{

    //! Constructor
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    XpetraOperator(const RCP<MueLu::Hierarchy<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& H) : Hierarchy_(H) { }
#else
    XpetraOperator(const RCP<MueLu::Hierarchy<Scalar, Node> >& H) : Hierarchy_(H) { }
#endif

    //! Destructor.
    virtual ~XpetraOperator() { }

    //@}

    //! Returns the Tpetra::Map object associated with the domain of this operator.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > getDomainMap() const {
      typedef Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> Matrix;
#else
    Teuchos::RCP<const Xpetra::Map<Node> > getDomainMap() const {
      typedef Xpetra::Matrix<Scalar, Node> Matrix;
#endif

      RCP<Matrix> A = Hierarchy_->GetLevel(0)->template Get<RCP<Matrix> >("A");
      return A->getDomainMap();
    }

    //! Returns the Tpetra::Map object associated with the range of this operator.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > getRangeMap() const {
      typedef Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> Matrix;
#else
    Teuchos::RCP<const Xpetra::Map<Node> > getRangeMap() const {
      typedef Xpetra::Matrix<Scalar, Node> Matrix;
#endif

      RCP<Matrix> A = Hierarchy_->GetLevel(0)->template Get<RCP<Matrix> >("A");
      return A->getRangeMap();
    }

    //! Returns in Y the result of a Xpetra::Operator applied to a Xpetra::MultiVector X.
    /*!
      \param[in]  X - Xpetra::MultiVector of dimension NumVectors to multiply with matrix.
      \param[out] Y - Xpetra::MultiVector of dimension NumVectors containing result.
    */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void apply(const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& X,
                                         Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Y,
#else
    void apply(const Xpetra::MultiVector<Scalar,Node>& X,
                                         Xpetra::MultiVector<Scalar,Node>& Y,
#endif
                                         Teuchos::ETransp /* mode */ = Teuchos::NO_TRANS,
                                         Scalar /* alpha */ = Teuchos::ScalarTraits<Scalar>::one(),
                                         Scalar /* beta */  = Teuchos::ScalarTraits<Scalar>::one()) const{
      try {
#ifdef HAVE_MUELU_DEBUG
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        typedef Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> Matrix;
#else
        typedef Xpetra::Matrix<Scalar, Node> Matrix;
#endif
        RCP<Matrix> A = Hierarchy_->GetLevel(0)->template Get<RCP<Matrix> >("A");

        // X is supposed to live in the range map of the operator (const rhs = B)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > Xop =
            Xpetra::MultiVectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(A->getRangeMap(),X.getNumVectors());
        RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > Yop =
            Xpetra::MultiVectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(A->getDomainMap(),Y.getNumVectors());
#else
        RCP<Xpetra::MultiVector<Scalar,Node> > Xop =
            Xpetra::MultiVectorFactory<Scalar,Node>::Build(A->getRangeMap(),X.getNumVectors());
        RCP<Xpetra::MultiVector<Scalar,Node> > Yop =
            Xpetra::MultiVectorFactory<Scalar,Node>::Build(A->getDomainMap(),Y.getNumVectors());
#endif
        TEUCHOS_TEST_FOR_EXCEPTION(A->getRangeMap()->isSameAs(*(Xop->getMap())) == false, std::logic_error,
                                   "MueLu::XpetraOperator::apply: map of X is incompatible with range map of A");
        TEUCHOS_TEST_FOR_EXCEPTION(A->getDomainMap()->isSameAs(*(Yop->getMap())) == false, std::logic_error,
                                   "MueLu::XpetraOperator::apply: map of Y is incompatible with domain map of A");
#endif

        Y.putScalar(Teuchos::ScalarTraits<Scalar>::zero());
        Hierarchy_->Iterate(X, Y, 1, true);
      } catch (std::exception& e) {
        //FIXME add message and rethrow
        std::cerr << "Caught an exception in MueLu::XpetraOperator::apply():" << std::endl
            << e.what() << std::endl;
      }
    }

    //! Indicates whether this operator supports applying the adjoint operator.
    bool hasTransposeApply() const { return false; }

    //! Compute a residual R = B - (*this) * X
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void residual(const Xpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > & X,
                  const Xpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > & B,
                  Xpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > & R) const {
#else
    void residual(const Xpetra::MultiVector< Scalar, Node > & X,
                  const Xpetra::MultiVector< Scalar, Node > & B,
                  Xpetra::MultiVector< Scalar, Node > & R) const {
#endif
      using STS = Teuchos::ScalarTraits<Scalar>;
      R.update(STS::one(),B,STS::zero());
      this->apply (X, R, Teuchos::NO_TRANS, -STS::one(), STS::one());   
    }      
    
    //! @name MueLu specific
    //@{

    //! Direct access to the underlying MueLu::Hierarchy.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<MueLu::Hierarchy<Scalar, LocalOrdinal, GlobalOrdinal, Node> > GetHierarchy() const { return Hierarchy_; }
#else
    RCP<MueLu::Hierarchy<Scalar, Node> > GetHierarchy() const { return Hierarchy_; }
#endif

    //@}

  private:
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<MueLu::Hierarchy<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Hierarchy_;
#else
    RCP<MueLu::Hierarchy<Scalar, Node> > Hierarchy_;
#endif
  };

} // namespace

#endif // MUELU_XPETRAOPERATOR_DECL_HPP
