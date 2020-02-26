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
#ifndef MUELU_SHIFTEDLAPLACIANOPERATOR_DECL_HPP
#define MUELU_SHIFTEDLAPLACIANOPERATOR_DECL_HPP

#include "MueLu_ConfigDefs.hpp"

#ifdef HAVE_MUELU_TPETRA
#include <Tpetra_Operator.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include "MueLu_Level.hpp"
#include "MueLu_Hierarchy_decl.hpp"
#include "MueLu_Utilities.hpp"

// TODO: Kokkos headers

namespace MueLu {

  /*! @brief Wraps an existing MueLu::Hierarchy as a Tpetra::Operator, with an optional two-level correction.
      Intended to be used with MueLu::ShiftedLaplacian.
  */
  template <class Scalar = Tpetra::Operator<>::scalar_type,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
            class LocalOrdinal = typename Tpetra::Operator<Scalar>::local_ordinal_type,
            class GlobalOrdinal = typename Tpetra::Operator<Scalar, LocalOrdinal>::global_ordinal_type,
            class Node = typename Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal>::node_type>
#else
            class Node = typename Tpetra::Operator<Scalar>::node_type>
#endif
  class ShiftedLaplacianOperator
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    : public Tpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node>
#else
    : public Tpetra::Operator<Scalar,Node>
#endif
  {

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>      Matrix;
    typedef Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>   CrsMatrix;
    typedef Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> MV;
    typedef Tpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node>    OP;
    typedef MueLu::Utilities<Scalar,LocalOrdinal,GlobalOrdinal,Node>        MUtils;
#else
    using LocalOrdinal = typename Tpetra::Map<>::local_ordinal_type;
    using GlobalOrdinal = typename Tpetra::Map<>::global_ordinal_type;
    typedef Xpetra::Matrix<Scalar,Node>      Matrix;
    typedef Tpetra::CrsMatrix<Scalar,Node>   CrsMatrix;
    typedef Tpetra::MultiVector<Scalar,Node> MV;
    typedef Tpetra::Operator<Scalar,Node>    OP;
    typedef MueLu::Utilities<Scalar,Node>        MUtils;
#endif
  public:

    //! @name Constructor/Destructor
    //@{

    //! Constructor
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    ShiftedLaplacianOperator(const RCP<MueLu::Hierarchy<Scalar, LocalOrdinal, GlobalOrdinal, Node> > & H) : Hierarchy_(H), option_(0) { }
#else
    ShiftedLaplacianOperator(const RCP<MueLu::Hierarchy<Scalar, Node> > & H) : Hierarchy_(H), option_(0) { }
#endif

    //! Auxiliary Constructor
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    ShiftedLaplacianOperator(const RCP<MueLu::Hierarchy<Scalar, LocalOrdinal, GlobalOrdinal, Node> > & H,
#else
    ShiftedLaplacianOperator(const RCP<MueLu::Hierarchy<Scalar, Node> > & H,
#endif
                             const RCP<Matrix> A, int cycles, int iters, int option, double tol) : Hierarchy_(H), A_(A), cycles_(cycles), iters_(iters), option_(option), tol_(tol)
    {

      // setup 2-level correction
      /*RCP< MueLu::Level > Level1 = H -> GetLevel(1);
      R_ = Level1 -> Get< RCP<Matrix> >("R");
      P_ = Level1 -> Get< RCP<Matrix> >("P");
      //RCP<Matrix> AP = Level1 -> Get< RCP<Matrix> >("AP graph");
      RCP<Matrix> AP;
      AP = MUtils::Multiply(*A_, false, *P_, false, AP);
      // Optimization storage option. If matrix is not changing later, allow this.
      bool doOptimizedStorage = true;
      // Reuse coarse matrix memory if available (multiple solve)
      //RCP<Matrix> Ac = Level1 -> Get< RCP<Matrix> >("RAP graph");
      RCP<Matrix> Ac;
      Ac = MUtils::Multiply(*R_, false, *AP, false, Ac, true, doOptimizedStorage);
      Ac_ = MUtils::Op2NonConstTpetraCrs(Ac);

      // Setup Belos for two-level correction
      BelosList_ = rcp( new Teuchos::ParameterList("GMRES") );
      BelosList_ -> set("Maximum Iterations", iters_ );
      BelosList_ -> set("Convergence Tolerance", tol_ );
      BelosLP_   = rcp( new Belos::LinearProblem<Scalar,MV,OP> );
      BelosLP_   -> setOperator ( Ac_ );
      BelosSM_   = rcp( new Belos::BlockGmresSolMgr<Scalar,MV,OP>(BelosLP_, BelosList_) );*/

    }



    //! Destructor.
    virtual ~ShiftedLaplacianOperator() { }

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
      \param[in] X - Tpetra::MultiVector of dimension NumVectors to multiply with matrix.
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
                                         Scalar beta = Teuchos::ScalarTraits<Scalar>::one()) const;

    //! Indicates whether this operator supports applying the adjoint operator.
    bool hasTransposeApply() const;

  private:

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<MueLu::Hierarchy<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Hierarchy_;
    RCP< Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > R_, P_, A_;
    RCP< Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > Ac_;
#else
    RCP<MueLu::Hierarchy<Scalar, Node> > Hierarchy_;
    RCP< Xpetra::Matrix<Scalar,Node> > R_, P_, A_;
    RCP< Tpetra::CrsMatrix<Scalar,Node> > Ac_;
#endif
    RCP< Teuchos::ParameterList >  BelosList_;

    //RCP< Belos::LinearProblem<Scalar,MV,OP> > BelosLP_;
    //RCP< Belos::SolverManager<Scalar,MV,OP> > BelosSM_;

    // cycles -> number of V-cycles
    // iters  -> number of GMRES iterations per correction
    // option -> 0 if no correction is desired
    int cycles_, iters_, option_;
    double tol_;

  };

} // namespace

#endif //ifdef HAVE_MUELU_TPETRA

#endif // MUELU_SHIFTEDLAPLACIANOPERATOR_DECL_HPP
