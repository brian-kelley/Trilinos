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
#ifndef MUELU_TPETRAOPERATOR_DECL_HPP
#define MUELU_TPETRAOPERATOR_DECL_HPP

#include "MueLu_ConfigDefs.hpp"

#ifdef HAVE_MUELU_TPETRA
#include <Tpetra_Operator.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include "MueLu_Level.hpp"
#include "MueLu_Hierarchy_decl.hpp"

namespace MueLu {

/*!  @brief Wraps an existing MueLu::Hierarchy as a Tpetra::Operator.
*/
  template <class Scalar = Tpetra::Operator<>::scalar_type,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            class LocalOrdinal = typename Tpetra::Operator<Scalar>::local_ordinal_type,
            class GlobalOrdinal = typename Tpetra::Operator<Scalar, LocalOrdinal>::global_ordinal_type,
            class Node = typename Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal>::node_type>
  class TpetraOperator : public Tpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node> {
#else
            class Node = typename Tpetra::Operator<Scalar>::node_type>
  class TpetraOperator : public Tpetra::Operator<Scalar,Node> {
#endif
  protected:
#ifndef TPETRA_ENABLE_TEMPLATE_ORDINALS
    using LocalOrdinal = typename Tpetra::Map<>::local_ordinal_type;
    using GlobalOrdinal = typename Tpetra::Map<>::global_ordinal_type;
#endif
    TpetraOperator() { }
  public:

    //! @name Constructor/Destructor
    //@{

    //! Constructor
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraOperator(const RCP<Xpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& Op) : Operator_(Op){ }
#else
    TpetraOperator(const RCP<Xpetra::Operator<Scalar, Node> >& Op) : Operator_(Op){ }
#endif

    //! Constructor
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraOperator(const RCP<MueLu::Hierarchy<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& H) : Hierarchy_(H){ }
#else
    TpetraOperator(const RCP<MueLu::Hierarchy<Scalar, Node> >& H) : Hierarchy_(H){ }
#endif

    //! Destructor.
    virtual ~TpetraOperator() { }

    //@}

    //! Returns the Tpetra::Map object associated with the domain of this operator.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > getDomainMap() const;
#else
    Teuchos::RCP<const Tpetra::Map<Node> > getDomainMap() const;
#endif

    //! Returns the Tpetra::Map object associated with the range of this operator.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > getRangeMap() const;
#else
    Teuchos::RCP<const Tpetra::Map<Node> > getRangeMap() const;
#endif

    //! Returns in Y the result of a Tpetra::Operator applied to a Tpetra::MultiVector X.
    /*!
      \param[in]  X - Tpetra::MultiVector of dimension NumVectors to multiply with matrix.
      \param[out] Y -Tpetra::MultiVector of dimension NumVectors containing result.
    */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void apply(const Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& X,
                                         Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Y,
#else
    void apply(const Tpetra::MultiVector<Scalar,Node>& X,
                                         Tpetra::MultiVector<Scalar,Node>& Y,
#endif
                                         Teuchos::ETransp mode = Teuchos::NO_TRANS,
                                         Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
                                         Scalar beta  = Teuchos::ScalarTraits<Scalar>::one()) const;

    //! Indicates whether this operator supports applying the adjoint operator.
    bool hasTransposeApply() const;

    //! @name MueLu specific
    //@{

    //! Direct access to the underlying MueLu::Hierarchy.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<MueLu::Hierarchy<Scalar, LocalOrdinal, GlobalOrdinal, Node> > GetHierarchy() const;
#else
    RCP<MueLu::Hierarchy<Scalar, Node> > GetHierarchy() const;
#endif

    //! Direct access to the underlying MueLu::Operator
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Xpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> > GetOperator() const;
#else
    RCP<Xpetra::Operator<Scalar, Node> > GetOperator() const;
#endif

    //@}

  private:
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<MueLu::Hierarchy<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Hierarchy_;
    RCP<Xpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Operator_;
#else
    RCP<MueLu::Hierarchy<Scalar, Node> > Hierarchy_;
    RCP<Xpetra::Operator<Scalar, Node> > Operator_;
#endif

  };

} // namespace

#endif //ifdef HAVE_MUELU_TPETRA

#endif // MUELU_TPETRAOPERATOR_DECL_HPP
