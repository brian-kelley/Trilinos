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
#ifndef BELOS_XPETRA_ADAPTER_MULTIVECTOR_HPP
#define BELOS_XPETRA_ADAPTER_MULTIVECTOR_HPP

#include <Xpetra_ConfigDefs.hpp>
#include <Xpetra_Exceptions.hpp>
#include <Xpetra_MultiVector.hpp>

#ifdef HAVE_XPETRA_EPETRA
#include <Xpetra_EpetraMultiVector.hpp>
#endif

#ifdef HAVE_XPETRA_TPETRA
#include <Xpetra_TpetraMultiVector.hpp>
#endif

#include <BelosConfigDefs.hpp>
#include <BelosTypes.hpp>
#include <BelosMultiVecTraits.hpp>
#include <BelosOperatorTraits.hpp>

#ifdef HAVE_XPETRA_EPETRA
#include <BelosEpetraAdapter.hpp>
#endif

#ifdef HAVE_XPETRA_TPETRA
#include <BelosTpetraAdapter.hpp>
#include <TpetraCore_config.h>
#endif

#ifdef HAVE_BELOS_TSQR
namespace BelosXpetraTsqrImpl {

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LO, class GO, class Node>
#else
  template<class Scalar, class Node>
#endif
  class XpetraStubTsqrAdaptor : public Teuchos::ParameterListAcceptorDefaultBase {
  public:
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Xpetra::MultiVector<Scalar, LO, GO, Node> MV;
#else
    using LO = typename Tpetra::Map<>::local_ordinal_type;
    using GO = typename Tpetra::Map<>::global_ordinal_type;
    typedef Xpetra::MultiVector<Scalar, Node> MV;
#endif
    typedef Scalar scalar_type;
    typedef LO ordinal_type;
    typedef Teuchos::SerialDenseMatrix<ordinal_type, scalar_type> dense_matrix_type;
    typedef typename Teuchos::ScalarTraits<scalar_type>::magnitudeType magnitude_type;

    XpetraStubTsqrAdaptor (const Teuchos::RCP<Teuchos::ParameterList>& /* plist */)
    {}

    XpetraStubTsqrAdaptor () {}

    Teuchos::RCP<const Teuchos::ParameterList>
    getValidParameters () const
    {
      return Teuchos::rcp (new Teuchos::ParameterList);
    }

    void
    setParameterList (const Teuchos::RCP<Teuchos::ParameterList>& /* plist */)
    {}

    void
    factorExplicit (MV& /* A */,
                    MV& /* Q */,
                    dense_matrix_type& /* R */,
                    const bool /* forceNonnegativeDiagonal */ = false)
    {
      TEUCHOS_TEST_FOR_EXCEPTION
        (true, std::logic_error, "Xpetra TSQR adaptor is only implemented "
         "for the Tpetra case");
    }

    int
    revealRank (MV& /* Q */,
                dense_matrix_type& /* R */,
                const magnitude_type& /* tol */)
    {
      TEUCHOS_TEST_FOR_EXCEPTION
        (true, std::logic_error, "Xpetra TSQR adaptor is only implemented "
         "for the Tpetra case");
    }
  };

#ifdef HAVE_XPETRA_TPETRA
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LO, class GO, class Node>
#else
  template<class Scalar, class Node>
#endif
  class XpetraTpetraTsqrAdaptor : public Teuchos::ParameterListAcceptorDefaultBase {
  public:
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Xpetra::MultiVector<Scalar, LO, GO, Node> MV;
#else
    using LO = typename Tpetra::Map<>::local_ordinal_type;
    using GO = typename Tpetra::Map<>::global_ordinal_type;
    typedef Xpetra::MultiVector<Scalar, Node> MV;
#endif
    typedef Scalar scalar_type;
    typedef LO ordinal_type;
    typedef Teuchos::SerialDenseMatrix<ordinal_type, scalar_type> dense_matrix_type;
    typedef typename Teuchos::ScalarTraits<scalar_type>::magnitudeType magnitude_type;

    XpetraTpetraTsqrAdaptor (const Teuchos::RCP<Teuchos::ParameterList>& plist)
      : tpetraImpl_ (plist)
    {}

    XpetraTpetraTsqrAdaptor () {}

    Teuchos::RCP<const Teuchos::ParameterList>
    getValidParameters () const
    {
      return tpetraImpl_.getValidParameters ();
    }

    void
    setParameterList (const Teuchos::RCP<Teuchos::ParameterList>& plist)
    {
      tpetraImpl_.setParameterList (plist);
    }

    void
    factorExplicit (MV& A,
                    MV& Q,
                    dense_matrix_type& R,
                    const bool forceNonnegativeDiagonal = false)
    {
      if (A.getMap()->lib() == Xpetra::UseTpetra) {
        tpetraImpl_.factorExplicit (toTpetra (A), toTpetra (Q), R, forceNonnegativeDiagonal);
        return;
      }
      XPETRA_FACTORY_ERROR_IF_EPETRA(A.getMap()->lib());
      XPETRA_FACTORY_END;
    }

    int
    revealRank (MV& Q,
                dense_matrix_type& R,
                const magnitude_type& tol)
    {
      if (Q.getMap()->lib() == Xpetra::UseTpetra) {
        return tpetraImpl_.revealRank (toTpetra (Q), R, tol);
      }
      XPETRA_FACTORY_ERROR_IF_EPETRA(Q.getMap()->lib());
      XPETRA_FACTORY_END;
    }

  private:
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef ::Tpetra::TsqrAdaptor< ::Tpetra::MultiVector<Scalar,LO,GO,Node> > tpetra_tsqr_adaptor_type;
#else
    typedef ::Tpetra::TsqrAdaptor< ::Tpetra::MultiVector<Scalar,Node> > tpetra_tsqr_adaptor_type;
#endif
    tpetra_tsqr_adaptor_type tpetraImpl_;
  };
#endif // HAVE_XPETRA_TPETRA

} // namespace BelosXpetraTsqrImpl
#endif // HAVE_BELOS_TSQR

namespace Belos {

  using Teuchos::RCP;
  using Teuchos::rcp;

  ////////////////////////////////////////////////////////////////////
  //
  // Implementation of the Belos::MultiVecTraits for Xpetra::MultiVector.
  //
  ////////////////////////////////////////////////////////////////////

  /*! \brief Template specialization of Belos::MultiVecTraits class using the Xpetra::MultiVector class.
    This interface will ensure that any Xpetra::MultiVector will be accepted by the Belos
    templated solvers.
  */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class Scalar, class LO, class GO, class Node>
  class MultiVecTraits<Scalar, Xpetra::MultiVector<Scalar,LO,GO,Node> > {
#else
  template<class Scalar, class Node>
  class MultiVecTraits<Scalar, Xpetra::MultiVector<Scalar,Node> > {
#endif
  private:
#ifdef HAVE_XPETRA_TPETRA
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Xpetra::TpetraMultiVector<Scalar,LO,GO,Node>                   TpetraMultiVector;
    typedef MultiVecTraits<Scalar,Tpetra::MultiVector<Scalar,LO,GO,Node> > MultiVecTraitsTpetra;
#else
    using LO = typename Tpetra::Map<>::local_ordinal_type;
    using GO = typename Tpetra::Map<>::global_ordinal_type;
    typedef Xpetra::TpetraMultiVector<Scalar,Node>                   TpetraMultiVector;
    typedef MultiVecTraits<Scalar,Tpetra::MultiVector<Scalar,Node> > MultiVecTraitsTpetra;
#endif
#endif

  public:

#ifdef HAVE_BELOS_XPETRA_TIMERS
    static RCP<Teuchos::Time> mvTimesMatAddMvTimer_, mvTransMvTimer_;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> > Clone( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const int numvecs )
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> > Clone( const Xpetra::MultiVector<Scalar,Node>& mv, const int numvecs )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::Clone(toTpetra(mv), numvecs)));
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> > CloneCopy( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> > CloneCopy( const Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneCopy(toTpetra(mv))));
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> > CloneCopy( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<int>& index )
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> > CloneCopy( const Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<int>& index )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneCopy(toTpetra(mv), index)));
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> >
    CloneCopy (const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv,
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> >
    CloneCopy (const Xpetra::MultiVector<Scalar,Node>& mv,
#endif
               const Teuchos::Range1D& index)
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneCopy(toTpetra(mv), index)));
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> > CloneViewNonConst( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<int>& index )
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> > CloneViewNonConst( Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<int>& index )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneViewNonConst(toTpetra(mv), index)));
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> >
    CloneViewNonConst(Xpetra::MultiVector<Scalar,LO,GO,Node>& mv,
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> >
    CloneViewNonConst(Xpetra::MultiVector<Scalar,Node>& mv,
#endif
                      const Teuchos::Range1D& index)
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneViewNonConst(toTpetra(mv), index)));
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<const Xpetra::MultiVector<Scalar,LO,GO,Node> > CloneView(const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<int>& index )
#else
    static RCP<const Xpetra::MultiVector<Scalar,Node> > CloneView(const Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<int>& index )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        //TODO: double check if the const_cast is safe here.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        RCP<const Tpetra::MultiVector<Scalar,LO,GO,Node> > r = MultiVecTraitsTpetra::CloneView(toTpetra(mv), index);
        return rcp(new TpetraMultiVector(Teuchos::rcp_const_cast<Tpetra::MultiVector<Scalar,LO,GO,Node> >(r)));
#else
        RCP<const Tpetra::MultiVector<Scalar,Node> > r = MultiVecTraitsTpetra::CloneView(toTpetra(mv), index);
        return rcp(new TpetraMultiVector(Teuchos::rcp_const_cast<Tpetra::MultiVector<Scalar,Node> >(r)));
#endif
      }
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<const Xpetra::MultiVector<Scalar,LO,GO,Node> >
    CloneView (const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv,
#else
    static RCP<const Xpetra::MultiVector<Scalar,Node> >
    CloneView (const Xpetra::MultiVector<Scalar,Node>& mv,
#endif
               const Teuchos::Range1D& index)
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        //TODO: double check if the const_cast is safe here.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        RCP<const Tpetra::MultiVector<Scalar,LO,GO,Node> > r = MultiVecTraitsTpetra::CloneView(toTpetra(mv), index);
        return rcp(new TpetraMultiVector(Teuchos::rcp_const_cast<Tpetra::MultiVector<Scalar,LO,GO,Node> >(r)));
#else
        RCP<const Tpetra::MultiVector<Scalar,Node> > r = MultiVecTraitsTpetra::CloneView(toTpetra(mv), index);
        return rcp(new TpetraMultiVector(Teuchos::rcp_const_cast<Tpetra::MultiVector<Scalar,Node> >(r)));
#endif
      }
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static ptrdiff_t GetGlobalLength( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static ptrdiff_t GetGlobalLength( const Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return MultiVecTraitsTpetra::GetGlobalLength(toTpetra(mv));
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static int GetNumberVecs( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static int GetNumberVecs( const Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return MultiVecTraitsTpetra::GetNumberVecs(toTpetra(mv));
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static bool HasConstantStride( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static bool HasConstantStride( const Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return  MultiVecTraitsTpetra::HasConstantStride(toTpetra(mv));
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvTimesMatAddMv( Scalar alpha, const Xpetra::MultiVector<Scalar,LO,GO,Node>& A,
#else
    static void MvTimesMatAddMv( Scalar alpha, const Xpetra::MultiVector<Scalar,Node>& A,
#endif
                                 const Teuchos::SerialDenseMatrix<int,Scalar>& B,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                                 Scalar beta, Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
                                 Scalar beta, Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_BELOS_XPETRA_TIMERS
      Teuchos::TimeMonitor lcltimer(*mvTimesMatAddMvTimer_);
#endif


#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvTimesMatAddMv(alpha, toTpetra(A), B, beta, toTpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvAddMv( Scalar alpha, const Xpetra::MultiVector<Scalar,LO,GO,Node>& A, Scalar beta, const Xpetra::MultiVector<Scalar,LO,GO,Node>& B, Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static void MvAddMv( Scalar alpha, const Xpetra::MultiVector<Scalar,Node>& A, Scalar beta, const Xpetra::MultiVector<Scalar,Node>& B, Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvAddMv(alpha, toTpetra(A), beta, toTpetra(B), toTpetra(mv));
        return;
      }

#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvScale ( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, Scalar alpha )
#else
    static void MvScale ( Xpetra::MultiVector<Scalar,Node>& mv, Scalar alpha )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvScale(toTpetra(mv), alpha);
        return;
      }
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvScale ( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<Scalar>& alphas )
#else
    static void MvScale ( Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<Scalar>& alphas )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvScale(toTpetra(mv), alphas);
        return;
      }
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvTransMv( Scalar alpha, const Xpetra::MultiVector<Scalar,LO,GO,Node>& A, const Xpetra::MultiVector<Scalar,LO,GO,Node>& B, Teuchos::SerialDenseMatrix<int,Scalar>& C)
#else
    static void MvTransMv( Scalar alpha, const Xpetra::MultiVector<Scalar,Node>& A, const Xpetra::MultiVector<Scalar,Node>& B, Teuchos::SerialDenseMatrix<int,Scalar>& C)
#endif
    {

#ifdef HAVE_BELOS_XPETRA_TIMERS
      Teuchos::TimeMonitor lcltimer(*mvTransMvTimer_);
#endif

#ifdef HAVE_XPETRA_TPETRA
      if (A.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvTransMv(alpha, toTpetra(A), toTpetra(B), C);
        return;
      }
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(A.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvDot( const Xpetra::MultiVector<Scalar,LO,GO,Node>& A, const Xpetra::MultiVector<Scalar,LO,GO,Node>& B, std::vector<Scalar> &dots)
#else
    static void MvDot( const Xpetra::MultiVector<Scalar,Node>& A, const Xpetra::MultiVector<Scalar,Node>& B, std::vector<Scalar> &dots)
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (A.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvDot(toTpetra(A), toTpetra(B), dots);
        return;
      }
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(A.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvNorm(const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, std::vector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType> &normvec, NormType type=TwoNorm)
#else
    static void MvNorm(const Xpetra::MultiVector<Scalar,Node>& mv, std::vector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType> &normvec, NormType type=TwoNorm)
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvNorm(toTpetra(mv), normvec, type);
        return;
      }
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void SetBlock( const Xpetra::MultiVector<Scalar,LO,GO,Node>& A, const std::vector<int>& index, Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static void SetBlock( const Xpetra::MultiVector<Scalar,Node>& A, const std::vector<int>& index, Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::SetBlock(toTpetra(A), index, toTpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

    static void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    SetBlock (const Xpetra::MultiVector<Scalar,LO,GO,Node>& A,
#else
    SetBlock (const Xpetra::MultiVector<Scalar,Node>& A,
#endif
              const Teuchos::Range1D& index,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
              Xpetra::MultiVector<Scalar,LO,GO,Node>& mv)
#else
              Xpetra::MultiVector<Scalar,Node>& mv)
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::SetBlock(toTpetra(A), index, toTpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

    static void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Assign (const Xpetra::MultiVector<Scalar,LO,GO,Node>& A,
            Xpetra::MultiVector<Scalar,LO,GO,Node>& mv)
#else
    Assign (const Xpetra::MultiVector<Scalar,Node>& A,
            Xpetra::MultiVector<Scalar,Node>& mv)
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::Assign(toTpetra(A), toTpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvRandom( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static void MvRandom( Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvRandom(toTpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvInit( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, Scalar alpha = Teuchos::ScalarTraits<Scalar>::zero() )
#else
    static void MvInit( Xpetra::MultiVector<Scalar,Node>& mv, Scalar alpha = Teuchos::ScalarTraits<Scalar>::zero() )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvInit(toTpetra(mv), alpha);
        return;
      }
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvPrint( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, std::ostream& os )
#else
    static void MvPrint( const Xpetra::MultiVector<Scalar,Node>& mv, std::ostream& os )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvPrint(toTpetra(mv), os);
        return;
      }
#endif

      XPETRA_FACTORY_ERROR_IF_EPETRA(mv.getMap()->lib());
      XPETRA_FACTORY_END;
    }

#ifdef HAVE_BELOS_TSQR
#  ifdef HAVE_XPETRA_TPETRA
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef BelosXpetraTsqrImpl::XpetraTpetraTsqrAdaptor<Scalar, LO, GO, Node> tsqr_adaptor_type;
#else
    typedef BelosXpetraTsqrImpl::XpetraTpetraTsqrAdaptor<Scalar, Node> tsqr_adaptor_type;
#endif
#  else
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef BelosXpetraTsqrImpl::XpetraStubTsqrAdaptor<Scalar, LO, GO, Node> tsqr_adaptor_type;
#else
    typedef BelosXpetraTsqrImpl::XpetraStubTsqrAdaptor<Scalar, Node> tsqr_adaptor_type;
#endif
#  endif // HAVE_XPETRA_TPETRA
#endif // HAVE_BELOS_TSQR
  };

#ifdef HAVE_XPETRA_EPETRA
#ifndef  EPETRA_NO_32BIT_GLOBAL_INDICES
  template<>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  class MultiVecTraits<double, Xpetra::MultiVector<double,int,int,Xpetra::EpetraNode> > {
#else
  class MultiVecTraits<double, Xpetra::MultiVector<double,Xpetra::EpetraNode> > {
#endif
  private:
    typedef double              Scalar;
    typedef int                 LO;
    typedef int                 GO;
    typedef Xpetra::EpetraNode  Node;

#ifdef HAVE_XPETRA_TPETRA // TODO check whether Tpetra is instantiated on all template parameters!
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Xpetra::TpetraMultiVector<Scalar,LO,GO,Node>                    TpetraMultiVector;
    typedef MultiVecTraits<Scalar, Tpetra::MultiVector<Scalar,LO,GO,Node> > MultiVecTraitsTpetra;
#else
    typedef Xpetra::TpetraMultiVector<Scalar,Node>                    TpetraMultiVector;
    typedef MultiVecTraits<Scalar, Tpetra::MultiVector<Scalar,Node> > MultiVecTraitsTpetra;
#endif
#endif
#endif

#ifdef HAVE_XPETRA_EPETRA
    typedef Xpetra::EpetraMultiVectorT<GO,Node>            EpetraMultiVector;
    typedef MultiVecTraits   <Scalar, Epetra_MultiVector>  MultiVecTraitsEpetra;
#endif

  public:

#ifdef HAVE_BELOS_XPETRA_TIMERS
    static RCP<Teuchos::Time> mvTimesMatAddMvTimer_, mvTransMvTimer_;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> > Clone( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const int numvecs )
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> > Clone( const Xpetra::MultiVector<Scalar,Node>& mv, const int numvecs )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::Clone(toTpetra(mv), numvecs)));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return rcp(new EpetraMultiVector(MultiVecTraitsEpetra::Clone(toEpetra(mv), numvecs)));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> > CloneCopy( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> > CloneCopy( const Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneCopy(toTpetra(mv))));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return rcp(new EpetraMultiVector(MultiVecTraitsEpetra::CloneCopy(toEpetra(mv))));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> > CloneCopy( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<int>& index )
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> > CloneCopy( const Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<int>& index )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneCopy(toTpetra(mv), index)));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return rcp(new EpetraMultiVector(MultiVecTraitsEpetra::CloneCopy(toEpetra(mv), index)));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> >
    CloneCopy (const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv,
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> >
    CloneCopy (const Xpetra::MultiVector<Scalar,Node>& mv,
#endif
               const Teuchos::Range1D& index)
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneCopy(toTpetra(mv), index)));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return rcp(new EpetraMultiVector(MultiVecTraitsEpetra::CloneCopy(toEpetra(mv), index)));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> > CloneViewNonConst( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<int>& index )
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> > CloneViewNonConst( Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<int>& index )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneViewNonConst(toTpetra(mv), index)));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return rcp(new EpetraMultiVector(MultiVecTraitsEpetra::CloneViewNonConst(toEpetra(mv), index)));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> >
    CloneViewNonConst(Xpetra::MultiVector<Scalar,LO,GO,Node>& mv,
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> >
    CloneViewNonConst(Xpetra::MultiVector<Scalar,Node>& mv,
#endif
                      const Teuchos::Range1D& index)
    {

#ifdef HAVE_MUELUA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneViewNonConst(toTpetra(mv), index)));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return rcp(new EpetraMultiVector(MultiVecTraitsEpetra::CloneViewNonConst(toEpetra(mv), index)));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<const Xpetra::MultiVector<Scalar,LO,GO,Node> > CloneView(const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<int>& index )
#else
    static RCP<const Xpetra::MultiVector<Scalar,Node> > CloneView(const Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<int>& index )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        //TODO: double check if the const_cast is safe here.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        RCP<const Tpetra::MultiVector<Scalar,LO,GO,Node> > r = MultiVecTraitsTpetra::CloneView(toTpetra(mv), index);
        return rcp(new TpetraMultiVector(Teuchos::rcp_const_cast<Tpetra::MultiVector<Scalar,LO,GO,Node> >(r)));
#else
        RCP<const Tpetra::MultiVector<Scalar,Node> > r = MultiVecTraitsTpetra::CloneView(toTpetra(mv), index);
        return rcp(new TpetraMultiVector(Teuchos::rcp_const_cast<Tpetra::MultiVector<Scalar,Node> >(r)));
#endif
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        //TODO: double check if the const_cast is safe here.
        RCP<const Epetra_MultiVector > r = MultiVecTraitsEpetra::CloneView(toEpetra(mv), index);
        return rcp(new EpetraMultiVector(Teuchos::rcp_const_cast<Epetra_MultiVector>(r)));
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<const Xpetra::MultiVector<Scalar,LO,GO,Node> >
    CloneView (const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv,
#else
    static RCP<const Xpetra::MultiVector<Scalar,Node> >
    CloneView (const Xpetra::MultiVector<Scalar,Node>& mv,
#endif
               const Teuchos::Range1D& index)
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        //TODO: double check if the const_cast is safe here.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        RCP<const Tpetra::MultiVector<Scalar,LO,GO,Node> > r = MultiVecTraitsTpetra::CloneView(toTpetra(mv), index);
        return rcp(new TpetraMultiVector(Teuchos::rcp_const_cast<Tpetra::MultiVector<Scalar,LO,GO,Node> >(r)));
#else
        RCP<const Tpetra::MultiVector<Scalar,Node> > r = MultiVecTraitsTpetra::CloneView(toTpetra(mv), index);
        return rcp(new TpetraMultiVector(Teuchos::rcp_const_cast<Tpetra::MultiVector<Scalar,Node> >(r)));
#endif
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
     if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        //TODO: double check if the const_cast is safe here.
        RCP<const Epetra_MultiVector > r = MultiVecTraitsEpetra::CloneView(toEpetra(mv), index);
        return rcp(new EpetraMultiVector(Teuchos::rcp_const_cast<Epetra_MultiVector>(r)));
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static ptrdiff_t GetGlobalLength( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static ptrdiff_t GetGlobalLength( const Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        return MultiVecTraitsTpetra::GetGlobalLength(toTpetra(mv));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return MultiVecTraitsEpetra::GetGlobalLength(toEpetra(mv));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static int GetNumberVecs( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static int GetNumberVecs( const Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        return MultiVecTraitsTpetra::GetNumberVecs(toTpetra(mv));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return MultiVecTraitsEpetra::GetNumberVecs(toEpetra(mv));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static bool HasConstantStride( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static bool HasConstantStride( const Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        return  MultiVecTraitsTpetra::HasConstantStride(toTpetra(mv));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return  MultiVecTraitsEpetra::HasConstantStride(toEpetra(mv));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvTimesMatAddMv( Scalar alpha, const Xpetra::MultiVector<Scalar,LO,GO,Node>& A,
#else
    static void MvTimesMatAddMv( Scalar alpha, const Xpetra::MultiVector<Scalar,Node>& A,
#endif
                                 const Teuchos::SerialDenseMatrix<int,Scalar>& B,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                                 Scalar beta, Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
                                 Scalar beta, Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_BELOS_XPETRA_TIMERS
      Teuchos::TimeMonitor lcltimer(*mvTimesMatAddMvTimer_);
#endif

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        MultiVecTraitsTpetra::MvTimesMatAddMv(alpha, toTpetra(A), B, beta, toTpetra(mv));
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvTimesMatAddMv(alpha, toEpetra(A), B, beta, toEpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvAddMv( Scalar alpha, const Xpetra::MultiVector<Scalar,LO,GO,Node>& A, Scalar beta, const Xpetra::MultiVector<Scalar,LO,GO,Node>& B, Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static void MvAddMv( Scalar alpha, const Xpetra::MultiVector<Scalar,Node>& A, Scalar beta, const Xpetra::MultiVector<Scalar,Node>& B, Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        MultiVecTraitsTpetra::MvAddMv(alpha, toTpetra(A), beta, toTpetra(B), toTpetra(mv));
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvAddMv(alpha, toEpetra(A), beta, toEpetra(B), toEpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvScale ( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, Scalar alpha )
#else
    static void MvScale ( Xpetra::MultiVector<Scalar,Node>& mv, Scalar alpha )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        MultiVecTraitsTpetra::MvScale(toTpetra(mv), alpha);
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvScale(toEpetra(mv), alpha);
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvScale ( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<Scalar>& alphas )
#else
    static void MvScale ( Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<Scalar>& alphas )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        MultiVecTraitsTpetra::MvScale(toTpetra(mv), alphas);
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvScale(toEpetra(mv), alphas);
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvTransMv( Scalar alpha, const Xpetra::MultiVector<Scalar,LO,GO,Node>& A, const Xpetra::MultiVector<Scalar,LO,GO,Node>& B, Teuchos::SerialDenseMatrix<int,Scalar>& C)
#else
    static void MvTransMv( Scalar alpha, const Xpetra::MultiVector<Scalar,Node>& A, const Xpetra::MultiVector<Scalar,Node>& B, Teuchos::SerialDenseMatrix<int,Scalar>& C)
#endif
    {

#ifdef HAVE_BELOS_XPETRA_TIMERS
      Teuchos::TimeMonitor lcltimer(*mvTransMvTimer_);
#endif

#ifdef HAVE_XPETRA_TPETRA
      if (A.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        MultiVecTraitsTpetra::MvTransMv(alpha, toTpetra(A), toTpetra(B), C);
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (A.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvTransMv(alpha, toEpetra(A), toEpetra(B), C);
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvDot( const Xpetra::MultiVector<Scalar,LO,GO,Node>& A, const Xpetra::MultiVector<Scalar,LO,GO,Node>& B, std::vector<Scalar> &dots)
#else
    static void MvDot( const Xpetra::MultiVector<Scalar,Node>& A, const Xpetra::MultiVector<Scalar,Node>& B, std::vector<Scalar> &dots)
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (A.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        MultiVecTraitsTpetra::MvDot(toTpetra(A), toTpetra(B), dots);
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (A.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvDot(toEpetra(A), toEpetra(B), dots);
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvNorm(const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, std::vector<Teuchos::ScalarTraits<Scalar>::magnitudeType> &normvec, NormType type=TwoNorm)
#else
    static void MvNorm(const Xpetra::MultiVector<Scalar,Node>& mv, std::vector<Teuchos::ScalarTraits<Scalar>::magnitudeType> &normvec, NormType type=TwoNorm)
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        MultiVecTraitsTpetra::MvNorm(toTpetra(mv), normvec, type);
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvNorm(toEpetra(mv), normvec, type);
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void SetBlock( const Xpetra::MultiVector<Scalar,LO,GO,Node>& A, const std::vector<int>& index, Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static void SetBlock( const Xpetra::MultiVector<Scalar,Node>& A, const std::vector<int>& index, Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        MultiVecTraitsTpetra::SetBlock(toTpetra(A), index, toTpetra(mv));
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::SetBlock(toEpetra(A), index, toEpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

    static void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    SetBlock (const Xpetra::MultiVector<Scalar,LO,GO,Node>& A,
#else
    SetBlock (const Xpetra::MultiVector<Scalar,Node>& A,
#endif
              const Teuchos::Range1D& index,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
              Xpetra::MultiVector<Scalar,LO,GO,Node>& mv)
#else
              Xpetra::MultiVector<Scalar,Node>& mv)
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        MultiVecTraitsTpetra::SetBlock(toTpetra(A), index, toTpetra(mv));
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::SetBlock(toEpetra(A), index, toEpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

    static void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Assign (const Xpetra::MultiVector<Scalar,LO,GO,Node>& A,
            Xpetra::MultiVector<Scalar,LO,GO,Node>& mv)
#else
    Assign (const Xpetra::MultiVector<Scalar,Node>& A,
            Xpetra::MultiVector<Scalar,Node>& mv)
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        MultiVecTraitsTpetra::Assign(toTpetra(A), toTpetra(mv));
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::Assign(toEpetra(A), toEpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvRandom( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static void MvRandom( Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        MultiVecTraitsTpetra::MvRandom(toTpetra(mv));
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvRandom(toEpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvInit( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, Scalar alpha = Teuchos::ScalarTraits<Scalar>::zero() )
#else
    static void MvInit( Xpetra::MultiVector<Scalar,Node>& mv, Scalar alpha = Teuchos::ScalarTraits<Scalar>::zero() )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        MultiVecTraitsTpetra::MvInit(toTpetra(mv), alpha);
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvInit(toEpetra(mv), alpha);
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvPrint( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, std::ostream& os )
#else
    static void MvPrint( const Xpetra::MultiVector<Scalar,Node>& mv, std::ostream& os )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
        MultiVecTraitsTpetra::MvPrint(toTpetra(mv), os);
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvPrint(toEpetra(mv), os);
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef HAVE_BELOS_TSQR
#  if defined(HAVE_XPETRA_TPETRA) && \
  ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
   (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef BelosXpetraTsqrImpl::XpetraTpetraTsqrAdaptor<Scalar, LO, GO, Node> tsqr_adaptor_type;
#else
    typedef BelosXpetraTsqrImpl::XpetraTpetraTsqrAdaptor<Scalar, Node> tsqr_adaptor_type;
#endif
#  else
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef BelosXpetraTsqrImpl::XpetraStubTsqrAdaptor<Scalar, LO, GO, Node> tsqr_adaptor_type;
#else
    typedef BelosXpetraTsqrImpl::XpetraStubTsqrAdaptor<Scalar, Node> tsqr_adaptor_type;
#endif
#  endif
#endif // HAVE_BELOS_TSQR
  };
#endif // #ifndef  EPETRA_NO_32BIT_GLOBAL_INDICES
#endif // HAVE_XPETRA_EPETRA


#ifdef HAVE_XPETRA_EPETRA
#ifndef  EPETRA_NO_64BIT_GLOBAL_INDICES
  template<>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  class MultiVecTraits<double, Xpetra::MultiVector<double,int,long long,Xpetra::EpetraNode> > {
#else
  class MultiVecTraits<double, Xpetra::MultiVector<double,Xpetra::EpetraNode> > {
#endif
  private:
    typedef double              Scalar;
    typedef int                 LO;
    typedef long long           GO;
    typedef Xpetra::EpetraNode  Node;

#ifdef HAVE_XPETRA_TPETRA // TODO check whether Tpetra is instantiated on all template parameters!
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Xpetra::TpetraMultiVector<Scalar,LO,GO,Node>                    TpetraMultiVector;
    typedef MultiVecTraits<Scalar, Tpetra::MultiVector<Scalar,LO,GO,Node> > MultiVecTraitsTpetra;
#else
    typedef Xpetra::TpetraMultiVector<Scalar,Node>                    TpetraMultiVector;
    typedef MultiVecTraits<Scalar, Tpetra::MultiVector<Scalar,Node> > MultiVecTraitsTpetra;
#endif
#endif
#endif

#ifdef HAVE_XPETRA_EPETRA
    typedef Xpetra::EpetraMultiVectorT<GO,Node>            EpetraMultiVector;
    typedef MultiVecTraits   <Scalar, Epetra_MultiVector>  MultiVecTraitsEpetra;
#endif

  public:

#ifdef HAVE_BELOS_XPETRA_TIMERS
    static RCP<Teuchos::Time> mvTimesMatAddMvTimer_, mvTransMvTimer_;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> > Clone( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const int numvecs )
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> > Clone( const Xpetra::MultiVector<Scalar,Node>& mv, const int numvecs )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::Clone(toTpetra(mv), numvecs)));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return rcp(new EpetraMultiVector(MultiVecTraitsEpetra::Clone(toEpetra(mv), numvecs)));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> > CloneCopy( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> > CloneCopy( const Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneCopy(toTpetra(mv))));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return rcp(new EpetraMultiVector(MultiVecTraitsEpetra::CloneCopy(toEpetra(mv))));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> > CloneCopy( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<int>& index )
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> > CloneCopy( const Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<int>& index )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneCopy(toTpetra(mv), index)));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return rcp(new EpetraMultiVector(MultiVecTraitsEpetra::CloneCopy(toEpetra(mv), index)));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> >
    CloneCopy (const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv,
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> >
    CloneCopy (const Xpetra::MultiVector<Scalar,Node>& mv,
#endif
               const Teuchos::Range1D& index)
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneCopy(toTpetra(mv), index)));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return rcp(new EpetraMultiVector(MultiVecTraitsEpetra::CloneCopy(toEpetra(mv), index)));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> > CloneViewNonConst( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<int>& index )
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> > CloneViewNonConst( Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<int>& index )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneViewNonConst(toTpetra(mv), index)));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return rcp(new EpetraMultiVector(MultiVecTraitsEpetra::CloneViewNonConst(toEpetra(mv), index)));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> >
    CloneViewNonConst(Xpetra::MultiVector<Scalar,LO,GO,Node>& mv,
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> >
    CloneViewNonConst(Xpetra::MultiVector<Scalar,Node>& mv,
#endif
                      const Teuchos::Range1D& index)
    {

#ifdef HAVE_MUELUA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneViewNonConst(toTpetra(mv), index)));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return rcp(new EpetraMultiVector(MultiVecTraitsEpetra::CloneViewNonConst(toEpetra(mv), index)));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<const Xpetra::MultiVector<Scalar,LO,GO,Node> > CloneView(const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<int>& index )
#else
    static RCP<const Xpetra::MultiVector<Scalar,Node> > CloneView(const Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<int>& index )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        //TODO: double check if the const_cast is safe here.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        RCP<const Tpetra::MultiVector<Scalar,LO,GO,Node> > r = MultiVecTraitsTpetra::CloneView(toTpetra(mv), index);
        return rcp(new TpetraMultiVector(Teuchos::rcp_const_cast<Tpetra::MultiVector<Scalar,LO,GO,Node> >(r)));
#else
        RCP<const Tpetra::MultiVector<Scalar,Node> > r = MultiVecTraitsTpetra::CloneView(toTpetra(mv), index);
        return rcp(new TpetraMultiVector(Teuchos::rcp_const_cast<Tpetra::MultiVector<Scalar,Node> >(r)));
#endif
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        //TODO: double check if the const_cast is safe here.
        RCP<const Epetra_MultiVector > r = MultiVecTraitsEpetra::CloneView(toEpetra(mv), index);
        return rcp(new EpetraMultiVector(Teuchos::rcp_const_cast<Epetra_MultiVector>(r)));
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<const Xpetra::MultiVector<Scalar,LO,GO,Node> >
    CloneView (const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv,
#else
    static RCP<const Xpetra::MultiVector<Scalar,Node> >
    CloneView (const Xpetra::MultiVector<Scalar,Node>& mv,
#endif
               const Teuchos::Range1D& index)
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        //TODO: double check if the const_cast is safe here.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        RCP<const Tpetra::MultiVector<Scalar,LO,GO,Node> > r = MultiVecTraitsTpetra::CloneView(toTpetra(mv), index);
        return rcp(new TpetraMultiVector(Teuchos::rcp_const_cast<Tpetra::MultiVector<Scalar,LO,GO,Node> >(r)));
#else
        RCP<const Tpetra::MultiVector<Scalar,Node> > r = MultiVecTraitsTpetra::CloneView(toTpetra(mv), index);
        return rcp(new TpetraMultiVector(Teuchos::rcp_const_cast<Tpetra::MultiVector<Scalar,Node> >(r)));
#endif
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
     if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        //TODO: double check if the const_cast is safe here.
        RCP<const Epetra_MultiVector > r = MultiVecTraitsEpetra::CloneView(toEpetra(mv), index);
        return rcp(new EpetraMultiVector(Teuchos::rcp_const_cast<Epetra_MultiVector>(r)));
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static ptrdiff_t GetGlobalLength( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static ptrdiff_t GetGlobalLength( const Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        return MultiVecTraitsTpetra::GetGlobalLength(toTpetra(mv));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return MultiVecTraitsEpetra::GetGlobalLength(toEpetra(mv));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static int GetNumberVecs( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static int GetNumberVecs( const Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        return MultiVecTraitsTpetra::GetNumberVecs(toTpetra(mv));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return MultiVecTraitsEpetra::GetNumberVecs(toEpetra(mv));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static bool HasConstantStride( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static bool HasConstantStride( const Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        return  MultiVecTraitsTpetra::HasConstantStride(toTpetra(mv));
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra)
        return  MultiVecTraitsEpetra::HasConstantStride(toEpetra(mv));
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvTimesMatAddMv( Scalar alpha, const Xpetra::MultiVector<Scalar,LO,GO,Node>& A,
#else
    static void MvTimesMatAddMv( Scalar alpha, const Xpetra::MultiVector<Scalar,Node>& A,
#endif
                                 const Teuchos::SerialDenseMatrix<int,Scalar>& B,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                                 Scalar beta, Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
                                 Scalar beta, Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_BELOS_XPETRA_TIMERS
      Teuchos::TimeMonitor lcltimer(*mvTimesMatAddMvTimer_);
#endif

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        MultiVecTraitsTpetra::MvTimesMatAddMv(alpha, toTpetra(A), B, beta, toTpetra(mv));
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvTimesMatAddMv(alpha, toEpetra(A), B, beta, toEpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvAddMv( Scalar alpha, const Xpetra::MultiVector<Scalar,LO,GO,Node>& A, Scalar beta, const Xpetra::MultiVector<Scalar,LO,GO,Node>& B, Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static void MvAddMv( Scalar alpha, const Xpetra::MultiVector<Scalar,Node>& A, Scalar beta, const Xpetra::MultiVector<Scalar,Node>& B, Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        MultiVecTraitsTpetra::MvAddMv(alpha, toTpetra(A), beta, toTpetra(B), toTpetra(mv));
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvAddMv(alpha, toEpetra(A), beta, toEpetra(B), toEpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvScale ( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, Scalar alpha )
#else
    static void MvScale ( Xpetra::MultiVector<Scalar,Node>& mv, Scalar alpha )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        MultiVecTraitsTpetra::MvScale(toTpetra(mv), alpha);
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvScale(toEpetra(mv), alpha);
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvScale ( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<Scalar>& alphas )
#else
    static void MvScale ( Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<Scalar>& alphas )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        MultiVecTraitsTpetra::MvScale(toTpetra(mv), alphas);
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvScale(toEpetra(mv), alphas);
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvTransMv( Scalar alpha, const Xpetra::MultiVector<Scalar,LO,GO,Node>& A, const Xpetra::MultiVector<Scalar,LO,GO,Node>& B, Teuchos::SerialDenseMatrix<int,Scalar>& C)
#else
    static void MvTransMv( Scalar alpha, const Xpetra::MultiVector<Scalar,Node>& A, const Xpetra::MultiVector<Scalar,Node>& B, Teuchos::SerialDenseMatrix<int,Scalar>& C)
#endif
    {

#ifdef HAVE_BELOS_XPETRA_TIMERS
      Teuchos::TimeMonitor lcltimer(*mvTransMvTimer_);
#endif

#ifdef HAVE_XPETRA_TPETRA
      if (A.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        MultiVecTraitsTpetra::MvTransMv(alpha, toTpetra(A), toTpetra(B), C);
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (A.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvTransMv(alpha, toEpetra(A), toEpetra(B), C);
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvDot( const Xpetra::MultiVector<Scalar,LO,GO,Node>& A, const Xpetra::MultiVector<Scalar,LO,GO,Node>& B, std::vector<Scalar> &dots)
#else
    static void MvDot( const Xpetra::MultiVector<Scalar,Node>& A, const Xpetra::MultiVector<Scalar,Node>& B, std::vector<Scalar> &dots)
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (A.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        MultiVecTraitsTpetra::MvDot(toTpetra(A), toTpetra(B), dots);
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (A.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvDot(toEpetra(A), toEpetra(B), dots);
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvNorm(const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, std::vector<Teuchos::ScalarTraits<Scalar>::magnitudeType> &normvec, NormType type=TwoNorm)
#else
    static void MvNorm(const Xpetra::MultiVector<Scalar,Node>& mv, std::vector<Teuchos::ScalarTraits<Scalar>::magnitudeType> &normvec, NormType type=TwoNorm)
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        MultiVecTraitsTpetra::MvNorm(toTpetra(mv), normvec, type);
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvNorm(toEpetra(mv), normvec, type);
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void SetBlock( const Xpetra::MultiVector<Scalar,LO,GO,Node>& A, const std::vector<int>& index, Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static void SetBlock( const Xpetra::MultiVector<Scalar,Node>& A, const std::vector<int>& index, Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        MultiVecTraitsTpetra::SetBlock(toTpetra(A), index, toTpetra(mv));
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::SetBlock(toEpetra(A), index, toEpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

    static void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    SetBlock (const Xpetra::MultiVector<Scalar,LO,GO,Node>& A,
#else
    SetBlock (const Xpetra::MultiVector<Scalar,Node>& A,
#endif
              const Teuchos::Range1D& index,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
              Xpetra::MultiVector<Scalar,LO,GO,Node>& mv)
#else
              Xpetra::MultiVector<Scalar,Node>& mv)
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        MultiVecTraitsTpetra::SetBlock(toTpetra(A), index, toTpetra(mv));
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::SetBlock(toEpetra(A), index, toEpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

    static void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Assign (const Xpetra::MultiVector<Scalar,LO,GO,Node>& A,
            Xpetra::MultiVector<Scalar,LO,GO,Node>& mv)
#else
    Assign (const Xpetra::MultiVector<Scalar,Node>& A,
            Xpetra::MultiVector<Scalar,Node>& mv)
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        MultiVecTraitsTpetra::Assign(toTpetra(A), toTpetra(mv));
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::Assign(toEpetra(A), toEpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvRandom( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static void MvRandom( Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        MultiVecTraitsTpetra::MvRandom(toTpetra(mv));
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvRandom(toEpetra(mv));
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvInit( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, Scalar alpha = Teuchos::ScalarTraits<Scalar>::zero() )
#else
    static void MvInit( Xpetra::MultiVector<Scalar,Node>& mv, Scalar alpha = Teuchos::ScalarTraits<Scalar>::zero() )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        MultiVecTraitsTpetra::MvInit(toTpetra(mv), alpha);
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvInit(toEpetra(mv), alpha);
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvPrint( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, std::ostream& os )
#else
    static void MvPrint( const Xpetra::MultiVector<Scalar,Node>& mv, std::ostream& os )
#endif
    {

#ifdef HAVE_XPETRA_TPETRA
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
        MultiVecTraitsTpetra::MvPrint(toTpetra(mv), os);
        return;
#endif
      }
#endif

#ifdef HAVE_XPETRA_EPETRA
      if (mv.getMap()->lib() == Xpetra::UseEpetra) {
        MultiVecTraitsEpetra::MvPrint(toEpetra(mv), os);
        return;
      }
#endif

      XPETRA_FACTORY_END;
    }

#ifdef HAVE_BELOS_TSQR
#  if defined(HAVE_XPETRA_TPETRA) && \
  ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
   (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef BelosXpetraTsqrImpl::XpetraTpetraTsqrAdaptor<Scalar, LO, GO, Node> tsqr_adaptor_type;
#else
    typedef BelosXpetraTsqrImpl::XpetraTpetraTsqrAdaptor<Scalar, Node> tsqr_adaptor_type;
#endif
#  else
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef BelosXpetraTsqrImpl::XpetraStubTsqrAdaptor<Scalar, LO, GO, Node> tsqr_adaptor_type;
#else
    typedef BelosXpetraTsqrImpl::XpetraStubTsqrAdaptor<Scalar, Node> tsqr_adaptor_type;
#endif
#  endif
#endif // HAVE_BELOS_TSQR

  };
#endif // #ifndef  EPETRA_NO_64BIT_GLOBAL_INDICES
#endif // HAVE_XPETRA_EPETRA


} // end of Belos namespace

#endif
