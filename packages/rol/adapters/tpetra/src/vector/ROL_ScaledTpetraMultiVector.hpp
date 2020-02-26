// @HEADER
// ************************************************************************
//
//               Rapid Optimization Library (ROL) Package
//                 Copyright (2014) Sandia Corporation
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
// Questions? Contact lead developers:
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER

#ifndef ROL_SCALEDTPETRAMULTIVECTOR_HPP
#define ROL_SCALEDTPETRAMULTIVECTOR_HPP

/** \class ROL::ScaledTpetraMultiVector
    \brief Implements the ROL::Vector interface for a Tpetra_MultiVector
           with scaled dot function.
    \author Created by Drew Kouri
*/

#include "ROL_TpetraMultiVector.hpp"

namespace ROL {

template <class Real,
          class LO=Tpetra::Map<>::local_ordinal_type, 
          class GO=Tpetra::Map<>::global_ordinal_type,
          class Node=Tpetra::Map<>::node_type >
class PrimalScaledTpetraMultiVector;

template <class Real,
          class LO=Tpetra::Map<>::local_ordinal_type, 
          class GO=Tpetra::Map<>::global_ordinal_type,
          class Node=Tpetra::Map<>::node_type >
class DualScaledTpetraMultiVector;

template <class Real, class LO, class GO, class Node>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
class PrimalScaledTpetraMultiVector : public TpetraMultiVector<Real,LO,GO,Node> {
#else
class PrimalScaledTpetraMultiVector : public TpetraMultiVector<Real,Node> {
#endif
  private:
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    const ROL::Ptr<const Tpetra::Vector<Real,LO,GO,Node> > scale_vec_;
#else
    const ROL::Ptr<const Tpetra::Vector<Real,Node> > scale_vec_;
#endif
    mutable ROL::Ptr<DualScaledTpetraMultiVector<Real> > dual_vec_;
    mutable bool isDualInitialized_;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void applyScaling(Tpetra::MultiVector<Real,LO,GO,Node> &out,
                const Tpetra::MultiVector<Real,LO,GO,Node> &in) const {
#else
    void applyScaling(Tpetra::MultiVector<Real,Node> &out,
                const Tpetra::MultiVector<Real,Node> &in) const {
#endif
      Real zero(0), one(1);
      out.elementWiseMultiply(one,*scale_vec_,in,zero);
    }

  public:
    virtual ~PrimalScaledTpetraMultiVector() {}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    PrimalScaledTpetraMultiVector(const ROL::Ptr<Tpetra::MultiVector<Real,LO,GO,Node> > &tpetra_vec,
                                  const ROL::Ptr<const Tpetra::Vector<Real,LO,GO,Node> > &scale_vec)
      : TpetraMultiVector<Real,LO,GO,Node>(tpetra_vec),
#else
    PrimalScaledTpetraMultiVector(const ROL::Ptr<Tpetra::MultiVector<Real,Node> > &tpetra_vec,
                                  const ROL::Ptr<const Tpetra::Vector<Real,Node> > &scale_vec)
      : TpetraMultiVector<Real,Node>(tpetra_vec),
#endif
        scale_vec_(scale_vec), isDualInitialized_(false) {}

    Real dot( const Vector<Real> &x ) const {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      ROL_TEST_FOR_EXCEPTION( (TpetraMultiVector<Real,LO,GO,Node>::dimension() != x.dimension()),
#else
      ROL_TEST_FOR_EXCEPTION( (TpetraMultiVector<Real,Node>::dimension() != x.dimension()),
#endif
                                  std::invalid_argument,
                                  "Error: Vectors must have the same dimension." );
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const Tpetra::MultiVector<Real,LO,GO,Node> &ex
        = *(dynamic_cast<const TpetraMultiVector<Real,LO,GO,Node>&>(x).getVector());
      const Tpetra::MultiVector<Real,LO,GO,Node> &ey
        = *(TpetraMultiVector<Real,LO,GO,Node>::getVector());
#else
      const Tpetra::MultiVector<Real,Node> &ex
        = *(dynamic_cast<const TpetraMultiVector<Real,Node>&>(x).getVector());
      const Tpetra::MultiVector<Real,Node> &ey
        = *(TpetraMultiVector<Real,Node>::getVector());
#endif
      size_t n = ey.getNumVectors();
      // Scale x with scale_vec_
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      Tpetra::MultiVector<Real,LO,GO,Node> wex(TpetraMultiVector<Real>::getMap(), n);
#else
      Tpetra::MultiVector<Real,Node> wex(TpetraMultiVector<Real>::getMap(), n);
#endif
      applyScaling(wex,ex);
      // Perform Euclidean dot between *this and scaled x for each vector
      Teuchos::Array<Real> val(n,0);
      ey.dot( wex, val.view(0,n) );
      // Combine dots for each vector to get a scalar
      Real xy(0);
      for (size_t i = 0; i < n; ++i) {
        xy += val[i];
      }
      return xy;
    }

    ROL::Ptr<Vector<Real> > clone() const {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const Tpetra::MultiVector<Real,LO,GO,Node> &ey
        = *(TpetraMultiVector<Real,LO,GO,Node>::getVector());
#else
      const Tpetra::MultiVector<Real,Node> &ey
        = *(TpetraMultiVector<Real,Node>::getVector());
#endif
      size_t n = ey.getNumVectors();
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      return ROL::makePtr<PrimalScaledTpetraMultiVector<Real,LO,GO,Node>>(
             ROL::makePtr<Tpetra::MultiVector<Real,LO,GO,Node>>(TpetraMultiVector<Real>::getMap(),n),
#else
      return ROL::makePtr<PrimalScaledTpetraMultiVector<Real,Node>>(
             ROL::makePtr<Tpetra::MultiVector<Real,Node>>(TpetraMultiVector<Real>::getMap(),n),
#endif
             scale_vec_);
    }

    const Vector<Real> & dual() const {
      if ( !isDualInitialized_ ) {
        // Create new memory for dual vector
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        size_t n = TpetraMultiVector<Real,LO,GO,Node>::getVector()->getNumVectors();
        dual_vec_ = ROL::makePtr<DualScaledTpetraMultiVector<Real,LO,GO,Node>>(
                    ROL::makePtr<Tpetra::MultiVector<Real,LO,GO,Node>>(TpetraMultiVector<Real>::getMap(),n),
#else
        size_t n = TpetraMultiVector<Real,Node>::getVector()->getNumVectors();
        dual_vec_ = ROL::makePtr<DualScaledTpetraMultiVector<Real,Node>>(
                    ROL::makePtr<Tpetra::MultiVector<Real,Node>>(TpetraMultiVector<Real>::getMap(),n),
#endif
                    scale_vec_);
        isDualInitialized_ = true;
      }
      // Scale *this with scale_vec_ and place in dual vector
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      applyScaling(*(dual_vec_->getVector()),*(TpetraMultiVector<Real,LO,GO,Node>::getVector()));
#else
      applyScaling(*(dual_vec_->getVector()),*(TpetraMultiVector<Real,Node>::getVector()));
#endif
      return *dual_vec_;
    }

}; // class PrimalScaledTpetraMultiVector

template <class Real, class LO, class GO, class Node>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
class DualScaledTpetraMultiVector : public TpetraMultiVector<Real,LO,GO,Node> {
#else
class DualScaledTpetraMultiVector : public TpetraMultiVector<Real,Node> {
#endif
  private:
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    const ROL::Ptr<const Tpetra::Vector<Real,LO,GO,Node> > scale_vec_;
#else
    const ROL::Ptr<const Tpetra::Vector<Real,Node> > scale_vec_;
#endif
    mutable ROL::Ptr<PrimalScaledTpetraMultiVector<Real> > primal_vec_;
    mutable bool isDualInitialized_;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void applyScaling(Tpetra::MultiVector<Real,LO,GO,Node> &out,
                const Tpetra::MultiVector<Real,LO,GO,Node> &in) const {
#else
    void applyScaling(Tpetra::MultiVector<Real,Node> &out,
                const Tpetra::MultiVector<Real,Node> &in) const {
#endif
      Real zero(0), one(1);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      Tpetra::Vector<Real,LO,GO,Node> W(scale_vec_->getMap());
#else
      Tpetra::Vector<Real,Node> W(scale_vec_->getMap());
#endif
      W.reciprocal(*scale_vec_);
      out.elementWiseMultiply(one,W,in,zero);
    }

  public:
    virtual ~DualScaledTpetraMultiVector() {}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    DualScaledTpetraMultiVector(const ROL::Ptr<Tpetra::MultiVector<Real,LO,GO,Node> > &tpetra_vec,
                                const ROL::Ptr<const Tpetra::Vector<Real,LO,GO,Node> > &scale_vec)
      : TpetraMultiVector<Real,LO,GO,Node>(tpetra_vec),
#else
    DualScaledTpetraMultiVector(const ROL::Ptr<Tpetra::MultiVector<Real,Node> > &tpetra_vec,
                                const ROL::Ptr<const Tpetra::Vector<Real,Node> > &scale_vec)
      : TpetraMultiVector<Real,Node>(tpetra_vec),
#endif
        scale_vec_(scale_vec), isDualInitialized_(false) {}

    Real dot( const Vector<Real> &x ) const {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      ROL_TEST_FOR_EXCEPTION( (TpetraMultiVector<Real,LO,GO,Node>::dimension() != x.dimension()),
#else
      ROL_TEST_FOR_EXCEPTION( (TpetraMultiVector<Real,Node>::dimension() != x.dimension()),
#endif
                                  std::invalid_argument,
                                  "Error: Vectors must have the same dimension." );
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const Tpetra::MultiVector<Real,LO,GO,Node> &ex
        = *(dynamic_cast<const TpetraMultiVector<Real,LO,GO,Node>&>(x).getVector());
      const Tpetra::MultiVector<Real,LO,GO,Node> &ey
#else
      const Tpetra::MultiVector<Real,Node> &ex
        = *(dynamic_cast<const TpetraMultiVector<Real,Node>&>(x).getVector());
      const Tpetra::MultiVector<Real,Node> &ey
#endif
        = *(TpetraMultiVector<Real>::getVector());
      size_t n = ey.getNumVectors();
      // Scale x with 1/scale_vec_
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      Tpetra::MultiVector<Real,LO,GO,Node> wex(TpetraMultiVector<Real>::getMap(), n);
#else
      Tpetra::MultiVector<Real,Node> wex(TpetraMultiVector<Real>::getMap(), n);
#endif
      applyScaling(wex,ex);
      // Perform Euclidean dot between *this and scaled x for each vector
      Teuchos::Array<Real> val(n,0);
      ey.dot( wex, val.view(0,n) );
      // Combine dots for each vector to get a scalar
      Real xy(0);
      for (size_t i = 0; i < n; ++i) {
        xy += val[i];
      }
      return xy;
    }

    ROL::Ptr<Vector<Real> > clone() const {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const Tpetra::MultiVector<Real,LO,GO,Node> &ey
        = *(TpetraMultiVector<Real,LO,GO,Node>::getVector());
#else
      const Tpetra::MultiVector<Real,Node> &ey
        = *(TpetraMultiVector<Real,Node>::getVector());
#endif
      size_t n = ey.getNumVectors();  
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      return ROL::makePtr<DualScaledTpetraMultiVector<Real,LO,GO,Node>>(
             ROL::makePtr<Tpetra::MultiVector<Real,LO,GO,Node>>(TpetraMultiVector<Real>::getMap(),n),
#else
      return ROL::makePtr<DualScaledTpetraMultiVector<Real,Node>>(
             ROL::makePtr<Tpetra::MultiVector<Real,Node>>(TpetraMultiVector<Real>::getMap(),n),
#endif
             scale_vec_);
    }

    const Vector<Real> & dual() const {
      if ( !isDualInitialized_ ) {
        // Create new memory for dual vector
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        size_t n = TpetraMultiVector<Real,LO,GO,Node>::getVector()->getNumVectors();
        primal_vec_ = ROL::makePtr<PrimalScaledTpetraMultiVector<Real,LO,GO,Node>>(
                      ROL::makePtr<Tpetra::MultiVector<Real,LO,GO,Node>>(TpetraMultiVector<Real>::getMap(),n),
#else
        size_t n = TpetraMultiVector<Real,Node>::getVector()->getNumVectors();
        primal_vec_ = ROL::makePtr<PrimalScaledTpetraMultiVector<Real,Node>>(
                      ROL::makePtr<Tpetra::MultiVector<Real,Node>>(TpetraMultiVector<Real>::getMap(),n),
#endif
                      scale_vec_);
        isDualInitialized_ = true;
      }
      // Scale *this with scale_vec_ and place in dual vector
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      applyScaling(*(primal_vec_->getVector()),*(TpetraMultiVector<Real,LO,GO,Node>::getVector()));
#else
      applyScaling(*(primal_vec_->getVector()),*(TpetraMultiVector<Real,Node>::getVector()));
#endif
      return *primal_vec_;
    }

}; // class DualScaledTpetraMultiVector

} // namespace ROL

#endif
