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
#ifndef BELOS_XPETRA_ADAPTER_MULTIVECTOR_MP_VECTOR_HPP
#define BELOS_XPETRA_ADAPTER_MULTIVECTOR_MP_VECTOR_HPP

#include "BelosXpetraAdapterMultiVector.hpp"
#include "Stokhos_Sacado_Kokkos_MP_Vector.hpp"
#include "Belos_TpetraAdapter_MP_Vector.hpp"

#ifdef HAVE_XPETRA_TPETRA

namespace Belos { // should be moved to Belos or Xpetra?

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
  template<class Storage, class LO, class GO, class Node>
  class MultiVecTraits<typename Storage::value_type,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                       Xpetra::MultiVector<Sacado::MP::Vector<Storage>,
                                           LO,GO,Node> > {
#else
                       Xpetra::MultiVector<Sacado::MP::Vector<Storage>,Node> > {
#endif
  public:
    typedef typename Storage::ordinal_type s_ordinal;
    typedef typename Storage::value_type BaseScalar;
    typedef Sacado::MP::Vector<Storage> Scalar;
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef typename Tpetra::MultiVector<Scalar,LO,GO,Node>::dot_type dot_type;
    typedef typename Tpetra::MultiVector<Scalar,LO,GO,Node>::mag_type mag_type;
#else
    typedef typename Tpetra::MultiVector<Scalar,Node>::dot_type dot_type;
    typedef typename Tpetra::MultiVector<Scalar,Node>::mag_type mag_type;
#endif

  private:

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Xpetra::TpetraMultiVector<Scalar,LO,GO,Node>                   TpetraMultiVector;
    typedef MultiVecTraits<dot_type,Tpetra::MultiVector<Scalar,LO,GO,Node> > MultiVecTraitsTpetra;
#else
    typedef Xpetra::TpetraMultiVector<Scalar,Node>                   TpetraMultiVector;
    typedef MultiVecTraits<dot_type,Tpetra::MultiVector<Scalar,Node> > MultiVecTraitsTpetra;
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
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::Clone(toTpetra(mv), numvecs)));
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> > CloneCopy( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> > CloneCopy( const Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneCopy(toTpetra(mv))));
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> > CloneCopy( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<int>& index )
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> > CloneCopy( const Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<int>& index )
#endif
    {
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneCopy(toTpetra(mv), index)));
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
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneCopy(toTpetra(mv), index)));
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<Xpetra::MultiVector<Scalar,LO,GO,Node> > CloneViewNonConst( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<int>& index )
#else
    static RCP<Xpetra::MultiVector<Scalar,Node> > CloneViewNonConst( Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<int>& index )
#endif
    {
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneViewNonConst(toTpetra(mv), index)));
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
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return rcp(new TpetraMultiVector(MultiVecTraitsTpetra::CloneViewNonConst(toTpetra(mv), index)));
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static RCP<const Xpetra::MultiVector<Scalar,LO,GO,Node> > CloneView(const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<int>& index )
#else
    static RCP<const Xpetra::MultiVector<Scalar,Node> > CloneView(const Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<int>& index )
#endif
    {
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
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static ptrdiff_t GetGlobalLength( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static ptrdiff_t GetGlobalLength( const Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return MultiVecTraitsTpetra::GetGlobalLength(toTpetra(mv));
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static int GetNumberVecs( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static int GetNumberVecs( const Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return MultiVecTraitsTpetra::GetNumberVecs(toTpetra(mv));
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static bool HasConstantStride( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static bool HasConstantStride( const Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {
      if (mv.getMap()->lib() == Xpetra::UseTpetra)
        return  MultiVecTraitsTpetra::HasConstantStride(toTpetra(mv));
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvTimesMatAddMv( dot_type alpha, const Xpetra::MultiVector<Scalar,LO,GO,Node>& A,
#else
    static void MvTimesMatAddMv( dot_type alpha, const Xpetra::MultiVector<Scalar,Node>& A,
#endif
                                 const Teuchos::SerialDenseMatrix<int,dot_type>& B,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                                 dot_type beta, Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
                                 dot_type beta, Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {
#ifdef HAVE_BELOS_XPETRA_TIMERS
      Teuchos::TimeMonitor lcltimer(*mvTimesMatAddMvTimer_);
#endif
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvTimesMatAddMv(alpha, toTpetra(A), B, beta, toTpetra(mv));
        return;
      }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvAddMv( Scalar alpha, const Xpetra::MultiVector<Scalar,LO,GO,Node>& A, Scalar beta, const Xpetra::MultiVector<Scalar,LO,GO,Node>& B, Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static void MvAddMv( Scalar alpha, const Xpetra::MultiVector<Scalar,Node>& A, Scalar beta, const Xpetra::MultiVector<Scalar,Node>& B, Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvAddMv(alpha, toTpetra(A), beta, toTpetra(B), toTpetra(mv));
        return;
      }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvScale ( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, Scalar alpha )
#else
    static void MvScale ( Xpetra::MultiVector<Scalar,Node>& mv, Scalar alpha )
#endif
    {
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvScale(toTpetra(mv), alpha);
        return;
      }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvScale ( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<BaseScalar>& alphas )
#else
    static void MvScale ( Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<BaseScalar>& alphas )
#endif
    {
      std::vector<Scalar> alphas_mp(alphas.size());
      const size_t sz = alphas.size();
      for (size_t i=0; i<sz; ++i)
        alphas_mp[i] = alphas[i];
      MvScale (mv, alphas_mp);
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvScale ( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, const std::vector<Scalar>& alphas )
#else
    static void MvScale ( Xpetra::MultiVector<Scalar,Node>& mv, const std::vector<Scalar>& alphas )
#endif
    {
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvScale(toTpetra(mv), alphas);
        return;
      }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvTransMv( dot_type alpha, const Xpetra::MultiVector<Scalar,LO,GO,Node>& A, const Xpetra::MultiVector<Scalar,LO,GO,Node>& B, Teuchos::SerialDenseMatrix<int,dot_type>& C)
#else
    static void MvTransMv( dot_type alpha, const Xpetra::MultiVector<Scalar,Node>& A, const Xpetra::MultiVector<Scalar,Node>& B, Teuchos::SerialDenseMatrix<int,dot_type>& C)
#endif
    {
#ifdef HAVE_BELOS_XPETRA_TIMERS
      Teuchos::TimeMonitor lcltimer(*mvTransMvTimer_);
#endif

      if (A.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvTransMv(alpha, toTpetra(A), toTpetra(B), C);
        return;
      }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvDot( const Xpetra::MultiVector<Scalar,LO,GO,Node>& A, const Xpetra::MultiVector<Scalar,LO,GO,Node>& B, std::vector<dot_type> &dots)
#else
    static void MvDot( const Xpetra::MultiVector<Scalar,Node>& A, const Xpetra::MultiVector<Scalar,Node>& B, std::vector<dot_type> &dots)
#endif
    {
      if (A.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvDot(toTpetra(A), toTpetra(B), dots);
        return;
      }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvNorm(const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, std::vector<mag_type> &normvec, NormType type=TwoNorm)
#else
    static void MvNorm(const Xpetra::MultiVector<Scalar,Node>& mv, std::vector<mag_type> &normvec, NormType type=TwoNorm)
#endif
    {
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvNorm(toTpetra(mv), normvec, type);
        return;
      }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void SetBlock( const Xpetra::MultiVector<Scalar,LO,GO,Node>& A, const std::vector<int>& index, Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static void SetBlock( const Xpetra::MultiVector<Scalar,Node>& A, const std::vector<int>& index, Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::SetBlock(toTpetra(A), index, toTpetra(mv));
        return;
      }
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
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::SetBlock(toTpetra(A), index, toTpetra(mv));
        return;
      }
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
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::Assign(toTpetra(A), toTpetra(mv));
        return;
      }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvRandom( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv )
#else
    static void MvRandom( Xpetra::MultiVector<Scalar,Node>& mv )
#endif
    {
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvRandom(toTpetra(mv));
        return;
      }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvInit( Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, Scalar alpha = Teuchos::ScalarTraits<Scalar>::zero() )
#else
    static void MvInit( Xpetra::MultiVector<Scalar,Node>& mv, Scalar alpha = Teuchos::ScalarTraits<Scalar>::zero() )
#endif
    {
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvInit(toTpetra(mv), alpha);
        return;
      }
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    static void MvPrint( const Xpetra::MultiVector<Scalar,LO,GO,Node>& mv, std::ostream& os )
#else
    static void MvPrint( const Xpetra::MultiVector<Scalar,Node>& mv, std::ostream& os )
#endif
    {
      if (mv.getMap()->lib() == Xpetra::UseTpetra) {
        MultiVecTraitsTpetra::MvPrint(toTpetra(mv), os);
        return;
      }
    }

  };

} // end of Belos namespace

#endif

#endif
