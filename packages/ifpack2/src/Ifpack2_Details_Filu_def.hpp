/*@HEADER
// ***********************************************************************
//
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//                 Copyright (2009) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
//@HEADER
*/

/// @file Ifpack2_filu_def.hpp

#ifndef __IFPACK2_FILU_DEF_HPP__ 
#define __IFPACK2_FILU_DEF_HPP__ 

#include "Ifpack2_Details_Filu_decl.hpp"
#include "Ifpack2_Details_CrsArrays.hpp"
#include <impl/Kokkos_Timer.hpp>
#include <shylu_fastilu.hpp>

namespace Ifpack2
{
namespace Details
{

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
Filu<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
template<typename Scalar, typename Node>
Filu<Scalar, Node>::
#endif
Filu(Teuchos::RCP<const TRowMatrix> A) :
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  FastILU_Base<Scalar, LocalOrdinal, GlobalOrdinal, Node>(A) {}
#else
  FastILU_Base<Scalar, Node>(A) {}
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
int Filu<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
template<typename Scalar, typename Node>
int Filu<Scalar, Node>::
#endif
getSweeps() const
{
  return localPrec_->getNFact();
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
int Filu<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
template<typename Scalar, typename Node>
int Filu<Scalar, Node>::
#endif
getNTrisol() const
{
  return localPrec_->getNTrisol();
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
void Filu<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
template<typename Scalar, typename Node>
void Filu<Scalar, Node>::
#endif
checkLocalILU() const
{
  localPrec_->checkILU();
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
void Filu<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
template<typename Scalar, typename Node>
void Filu<Scalar, Node>::
#endif
checkLocalIC() const
{
  localPrec_->checkIC();
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
void Filu<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
template<typename Scalar, typename Node>
void Filu<Scalar, Node>::
#endif
initLocalPrec()
{
  auto nRows = this->mat_->getNodeNumRows();
  auto& p = this->params_;
  localPrec_ = Teuchos::rcp(new LocalFILU(this->localRowPtrs_, this->localColInds_, this->localValues_, nRows,
        p.nFact, p.nTrisol, p.level, p.omega,
        p.shift, p.guessFlag ? 1 : 0, p.blockSize));
  localPrec_->initialize();
  this->initTime_ = localPrec_->getInitializeTime();
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
void Filu<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
template<typename Scalar, typename Node>
void Filu<Scalar, Node>::
#endif
computeLocalPrec()
{
  //update values in local prec (until compute(), values aren't needed)
  localPrec_->setValues(this->localValues_);
  localPrec_->compute();
  this->computeTime_ = localPrec_->getComputeTime();
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
void Filu<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
template<typename Scalar, typename Node>
void Filu<Scalar, Node>::
#endif
applyLocalPrec(ScalarArray x, ScalarArray y) const
{
  localPrec_->apply(x, y);
  //since this may be applied to multiple vectors, add to applyTime_ instead of setting it
  this->applyTime_ += localPrec_->getApplyTime();
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
std::string Filu<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
template<typename Scalar, typename Node>
std::string Filu<Scalar, Node>::
#endif
getName() const
{
  return "Filu";
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
#define IFPACK2_DETAILS_FILU_INSTANT(S, L, G, N) \
template class Ifpack2::Details::Filu<S, L, G, N>;
#else
#define IFPACK2_DETAILS_FILU_INSTANT(S, N) \
template class Ifpack2::Details::Filu<S, N>;
#endif

} //namespace Details
} //namespace Ifpack2

#endif

