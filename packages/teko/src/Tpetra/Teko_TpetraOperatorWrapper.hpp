/*
// @HEADER
//
// ***********************************************************************
//
//      Teko: A package for block and physics based preconditioning
//                  Copyright 2010 Sandia Corporation
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
// Questions? Contact Eric C. Cyr (eccyr@sandia.gov)
//
// ***********************************************************************
//
// @HEADER

*/

#ifndef __Teko_TpetraOperatorWrapper_hpp__
#define __Teko_TpetraOperatorWrapper_hpp__

#include "Thyra_LinearOpBase.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"
#include "Teko_ConfigDefs.hpp"

#include <string>


namespace Teko {
namespace TpetraHelpers {
  using Teuchos::RCP;

  class TpetraOperatorWrapper;

  /// Abstract Mapping strategy for an TpetraOperatorWrapper
  class MappingStrategy {
  public:
     virtual ~MappingStrategy() {}

     /** \brief Copy an Epetra_MultiVector into a Thyra::MultiVectorBase
       *
       * Copy an Epetra_MultiVector into a Thyra::MultiVectorBase. The exact
       * method for copying is specified by the concrete implementations.
       *
       * \param[in]     epetraX Vector to be copied into the Thyra object
       * \param[in,out] thyraX  Destination Thyra object
       */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
     virtual void copyTpetraIntoThyra(const Tpetra::MultiVector<ST,LO,GO,NT>& tpetraX,
#else
     virtual void copyTpetraIntoThyra(const Tpetra::MultiVector<ST,NT>& tpetraX,
#endif
                                      const Teuchos::Ptr<Thyra::MultiVectorBase<ST> > & thyraX) const = 0;
                                      // const TpetraOperatorWrapper & eow) const = 0;

     /** \brief Copy an Thyra::MultiVectorBase into a Epetra_MultiVector
       *
       * Copy an Thyra::MultiVectorBase into an Epetra_MultiVector. The exact
       * method for copying is specified by the concrete implementations.
       *
       * \param[in]     thyraX  Source Thyra object
       * \param[in,out] epetraX Destination Epetra object
       */
     virtual void copyThyraIntoTpetra(const RCP<const Thyra::MultiVectorBase<ST> > & thyraX,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                                      Tpetra::MultiVector<ST,LO,GO,NT>& tpetraX) const = 0;
#else
                                      Tpetra::MultiVector<ST,NT>& tpetraX) const = 0;
#endif
                                      // const TpetraOperatorWrapper & eow) const = 0;

     /** \brief Domain map for this strategy */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
     virtual const RCP<const Tpetra::Map<LO,GO,NT> > domainMap() const = 0;
#else
     virtual const RCP<const Tpetra::Map<NT> > domainMap() const = 0;
#endif

     /** \brief Range map for this strategy */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
     virtual const RCP<const Tpetra::Map<LO,GO,NT> > rangeMap() const = 0;
#else
     virtual const RCP<const Tpetra::Map<NT> > rangeMap() const = 0;
#endif

     /** \brief Identifier string */
     virtual std::string toString() const = 0;
  };

  /// Flip a mapping strategy object around to give the "inverse" mapping strategy.
  class InverseMappingStrategy : public MappingStrategy {
  public:
     /** \brief Constructor to build a inverse MappingStrategy from
       * a forward map.
       */
     InverseMappingStrategy(const RCP<const MappingStrategy> & forward)
        : forwardStrategy_(forward)
     { }

     virtual ~InverseMappingStrategy() {}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
     virtual void copyTpetraIntoThyra(const Tpetra::MultiVector<ST,LO,GO,NT>& tpetraX,
#else
     virtual void copyTpetraIntoThyra(const Tpetra::MultiVector<ST,NT>& tpetraX,
#endif
                                      const Teuchos::Ptr<Thyra::MultiVectorBase<ST> > & thyraX) const
                                      // const TpetraOperatorWrapper & eow) const
     { forwardStrategy_->copyTpetraIntoThyra(tpetraX,thyraX); }

     virtual void copyThyraIntoTpetra(const RCP<const Thyra::MultiVectorBase<ST> > & thyraX,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                                      Tpetra::MultiVector<ST,LO,GO,NT>& tpetraX) const
#else
                                      Tpetra::MultiVector<ST,NT>& tpetraX) const
#endif
                                      // const TpetraOperatorWrapper & eow) const
     { forwardStrategy_->copyThyraIntoTpetra(thyraX,tpetraX); }

     /** \brief Domain map for this strategy */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
     virtual const RCP<const Tpetra::Map<LO,GO,NT> > domainMap() const
#else
     virtual const RCP<const Tpetra::Map<NT> > domainMap() const
#endif
     { return forwardStrategy_->rangeMap(); }

     /** \brief Range map for this strategy */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
     virtual const RCP<const Tpetra::Map<LO,GO,NT> > rangeMap() const
#else
     virtual const RCP<const Tpetra::Map<NT> > rangeMap() const
#endif
     { return forwardStrategy_->domainMap(); }

     /** \brief Identifier string */
     virtual std::string toString() const
     { return std::string("InverseMapping(")+forwardStrategy_->toString()+std::string(")"); }
  protected:
     /** \brief Forward mapping strategy object */
     const RCP<const MappingStrategy> forwardStrategy_;

  private:
     InverseMappingStrategy();
     InverseMappingStrategy(const InverseMappingStrategy &);
  };

  /// default mapping strategy for the basic TpetraOperatorWrapper
  class DefaultMappingStrategy : public MappingStrategy {
  public:
     /** */
     DefaultMappingStrategy(const RCP<const Thyra::LinearOpBase<ST> > & thyraOp,const Teuchos::Comm<Thyra::Ordinal> & comm);

     virtual ~DefaultMappingStrategy() {}

     /** \brief Copy an Epetra_MultiVector into a Thyra::MultiVectorBase
       *
       * Copy an Epetra_MultiVector into a Thyra::MultiVectorBase. The exact
       * method for copying is specified by the concrete implementations.
       *
       * \param[in]     epetraX Vector to be copied into the Thyra object
       * \param[in,out] thyraX  Destination Thyra object
       */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
     virtual void copyTpetraIntoThyra(const Tpetra::MultiVector<ST,LO,GO,NT>& tpetraX,
#else
     virtual void copyTpetraIntoThyra(const Tpetra::MultiVector<ST,NT>& tpetraX,
#endif
                                      const Teuchos::Ptr<Thyra::MultiVectorBase<ST> > & thyraX) const;
                                      // const TpetraOperatorWrapper & eow) const;

     /** \brief Copy an Thyra::MultiVectorBase into a Epetra_MultiVector
       *
       * Copy an Thyra::MultiVectorBase into an Epetra_MultiVector. The exact
       * method for copying is specified by the concrete implementations.
       *
       * \param[in]     thyraX  Source Thyra object
       * \param[in,out] epetraX Destination Epetra object
       */
     virtual void copyThyraIntoTpetra(const RCP<const Thyra::MultiVectorBase<ST> > & thyraX,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                                      Tpetra::MultiVector<ST,LO,GO,NT>& tpetraX) const;
#else
                                      Tpetra::MultiVector<ST,NT>& tpetraX) const;
#endif
                                      // const TpetraOperatorWrapper & eow) const;

     /** \brief Domain map for this strategy */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
     virtual const RCP<const Tpetra::Map<LO,GO,NT> > domainMap() const { return domainMap_; }
#else
     virtual const RCP<const Tpetra::Map<NT> > domainMap() const { return domainMap_; }
#endif

     /** \brief Range map for this strategy */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
     virtual const RCP<const Tpetra::Map<LO,GO,NT> > rangeMap() const { return rangeMap_; }
#else
     virtual const RCP<const Tpetra::Map<NT> > rangeMap() const { return rangeMap_; }
#endif

     /** \brief Identifier string */
     virtual std::string toString() const
     { return std::string("DefaultMappingStrategy"); }

  protected:
     RCP<const Thyra::VectorSpaceBase<ST> > domainSpace_; ///< Domain space object
     RCP<const Thyra::VectorSpaceBase<ST> > rangeSpace_; ///< Range space object

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
     RCP<const Tpetra::Map<LO,GO,NT> > domainMap_; ///< Pointer to the constructed domain map
     RCP<const Tpetra::Map<LO,GO,NT> > rangeMap_; ///< Pointer to the constructed range map
#else
     RCP<const Tpetra::Map<NT> > domainMap_; ///< Pointer to the constructed domain map
     RCP<const Tpetra::Map<NT> > rangeMap_; ///< Pointer to the constructed range map
#endif
  };

  /** \brief
   * Implements the Epetra_Operator interface with a Thyra LinearOperator. This
   * enables the use of absrtact Thyra operators in AztecOO as preconditioners and
   * operators, without being rendered into concrete Epetra matrices. This is my own
   * modified version that was originally in Thyra.
   */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  class TpetraOperatorWrapper : public Tpetra::Operator<ST,LO,GO,NT>
#else
  class TpetraOperatorWrapper : public Tpetra::Operator<ST,NT>
#endif
  {
  public:
    /** */
    TpetraOperatorWrapper(const RCP<const Thyra::LinearOpBase<ST> > & thyraOp);
    TpetraOperatorWrapper(const RCP<const Thyra::LinearOpBase<ST> > & thyraOp,
                          const RCP<const MappingStrategy> & mapStrategy);
    TpetraOperatorWrapper(const RCP<const MappingStrategy> & mapStrategy);

    /** */
    virtual ~TpetraOperatorWrapper() {;}

    /** */
    int SetUseTranspose(bool useTranspose) {
      useTranspose_ = useTranspose;
      return 0;
    }

    /** */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void apply(const Tpetra::MultiVector<ST,LO,GO,NT>& X, Tpetra::MultiVector<ST,LO,GO,NT>& Y, Teuchos::ETransp mode=Teuchos::NO_TRANS, ST alpha=Teuchos::ScalarTraits< ST >::one(), ST beta=Teuchos::ScalarTraits< ST >::zero()) const ;
#else
    void apply(const Tpetra::MultiVector<ST,NT>& X, Tpetra::MultiVector<ST,NT>& Y, Teuchos::ETransp mode=Teuchos::NO_TRANS, ST alpha=Teuchos::ScalarTraits< ST >::one(), ST beta=Teuchos::ScalarTraits< ST >::zero()) const ;
#endif

    /** */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    void applyInverse(const Tpetra::MultiVector<ST,LO,GO,NT>& X, Tpetra::MultiVector<ST,LO,GO,NT>& Y, Teuchos::ETransp mode=Teuchos::NO_TRANS, ST alpha=Teuchos::ScalarTraits< ST >::one(), ST beta=Teuchos::ScalarTraits< ST >::zero()) const ;
#else
    void applyInverse(const Tpetra::MultiVector<ST,NT>& X, Tpetra::MultiVector<ST,NT>& Y, Teuchos::ETransp mode=Teuchos::NO_TRANS, ST alpha=Teuchos::ScalarTraits< ST >::one(), ST beta=Teuchos::ScalarTraits< ST >::zero()) const ;
#endif

    /** */
    double NormInf() const ;

    /** */
    const char* Label() const {return label_.c_str();}

    /** */
    bool UseTranspose() const {return useTranspose_;}

    /** */
    bool HasNormInf() const {return false;}

    /** */
    const Teuchos::RCP<const Teuchos::Comm<Thyra::Ordinal> > & Comm() const {return comm_;}

    /** */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP<const Tpetra::Map<LO,GO,NT> > getDomainMap() const {return mapStrategy_->domainMap();}
#else
    Teuchos::RCP<const Tpetra::Map<NT> > getDomainMap() const {return mapStrategy_->domainMap();}
#endif

    /** */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP<const Tpetra::Map<LO,GO,NT> > getRangeMap() const {return mapStrategy_->rangeMap();}
#else
    Teuchos::RCP<const Tpetra::Map<NT> > getRangeMap() const {return mapStrategy_->rangeMap();}
#endif

    //! Return the thyra operator associated with this wrapper
    const RCP<const Thyra::LinearOpBase<ST> > getThyraOp() const
    { return thyraOp_; }

    //! Get the mapping strategy for this wrapper (translate between Thyra and Epetra)
    const RCP<const MappingStrategy> getMapStrategy() const
    { return mapStrategy_; }

    //! Get the number of block rows in this operator
    virtual int GetBlockRowCount();

    //! Get the number of block columns in this operator
    virtual int GetBlockColCount();

    //! Grab the i,j block
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP<const Tpetra::Operator<ST,LO,GO,NT> > GetBlock(int i,int j) const;
#else
    Teuchos::RCP<const Tpetra::Operator<ST,NT> > GetBlock(int i,int j) const;
#endif

  protected:
    /** */
    TpetraOperatorWrapper();

    /** */
    RCP<const Teuchos::Comm<Thyra::Ordinal> > getThyraComm(const Thyra::LinearOpBase<ST> & inOp) const;

    /** */
    void SetOperator(const RCP<const Thyra::LinearOpBase<ST> > & thyraOp,bool buildMap=true);

    /** */
    void SetMapStrategy(const RCP<const MappingStrategy> & mapStrategy)
    { mapStrategy_ = mapStrategy; }

    /** */
    RCP<const MappingStrategy> mapStrategy_;

    /** */
    RCP<const Thyra::LinearOpBase<ST> > thyraOp_;

    /** */
    bool useTranspose_;

    /** */
    RCP<const Teuchos::Comm<Thyra::Ordinal> > comm_;

    /** */
    std::string label_;
  };
} // end namespace Tpetra
} // end namespace Teko

#endif
