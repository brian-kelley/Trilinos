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

#include "Teko_TpetraHelpers.hpp"

// Thyra Includes
#include "Thyra_EpetraLinearOp.hpp"
#include "Thyra_BlockedLinearOpBase.hpp"
#include "Thyra_DefaultMultipliedLinearOp.hpp"
#include "Thyra_DefaultDiagonalLinearOp.hpp"
#include "Thyra_DefaultZeroLinearOp.hpp"
#include "Thyra_DefaultBlockedLinearOp.hpp"
#include "Thyra_EpetraThyraWrappers.hpp"
#include "Thyra_SpmdVectorBase.hpp"
#include "Thyra_SpmdVectorSpaceBase.hpp"
#include "Thyra_ScalarProdVectorSpaceBase.hpp"

// Epetra includes
#include "Epetra_Vector.h"

// EpetraExt includes
#include "EpetraExt_ProductOperator.h"
#include "EpetraExt_MatrixMatrix.h"

// Teko includes
#include "Teko_EpetraOperatorWrapper.hpp"
#include "Teko_Utilities.hpp"

// Tpetra
#include "Thyra_TpetraLinearOp.hpp" 
#include "Thyra_TpetraMultiVector.hpp" 
#include "Tpetra_CrsMatrix.hpp" 
#include "Tpetra_Vector.hpp" 
#include "Thyra_TpetraThyraWrappers.hpp" 
#include "TpetraExt_MatrixMatrix.hpp" 

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcpFromRef;
using Teuchos::rcp_dynamic_cast;
using Teuchos::null;

namespace Teko {
namespace TpetraHelpers {

/** \brief Convert a Tpetra_Vector into a diagonal linear operator.
  *
  * Convert a Tpetra_Vector into a diagonal linear operator. 
  *
  * \param[in] tv  Tpetra_Vector to use as the diagonal
  * \param[in] map Map related to the Tpetra_Vector
  * \param[in] lbl String to easily label the operator
  *
  * \returns A diagonal linear operator using the vector
  */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
const Teuchos::RCP<const Thyra::LinearOpBase<ST> > thyraDiagOp(const RCP<const Tpetra::Vector<ST,LO,GO,NT> > & tv,const Tpetra::Map<LO,GO,NT> & map,
#else
const Teuchos::RCP<const Thyra::LinearOpBase<ST> > thyraDiagOp(const RCP<const Tpetra::Vector<ST,NT> > & tv,const Tpetra::Map<NT> & map,
#endif
                                                                   const std::string & lbl)
{
   const RCP<const Thyra::VectorBase<ST> > thyraVec  // need a Thyra::VectorBase object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
         = Thyra::createConstVector<ST,LO,GO,NT>(tv,Thyra::createVectorSpace<ST,LO,GO,NT>(rcpFromRef(map)));
#else
         = Thyra::createConstVector<ST,NT>(tv,Thyra::createVectorSpace<ST,NT>(rcpFromRef(map)));
#endif
   Teuchos::RCP<Thyra::LinearOpBase<ST> > op 
         = Teuchos::rcp(new Thyra::DefaultDiagonalLinearOp<ST>(thyraVec));
   op->setObjectLabel(lbl);
   return op;
}

/** \brief Convert a Tpetra_Vector into a diagonal linear operator.
  *
  * Convert a Tpetra_Vector into a diagonal linear operator. 
  *
  * \param[in] tv  Tpetra_Vector to use as the diagonal
  * \param[in] map Map related to the Tpetra_Vector
  * \param[in] lbl String to easily label the operator
  *
  * \returns A diagonal linear operator using the vector
  */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
const Teuchos::RCP<Thyra::LinearOpBase<ST> > thyraDiagOp(const RCP<Tpetra::Vector<ST,LO,GO,NT> > & tv,const Tpetra::Map<LO,GO,NT> & map,
#else
const Teuchos::RCP<Thyra::LinearOpBase<ST> > thyraDiagOp(const RCP<Tpetra::Vector<ST,NT> > & tv,const Tpetra::Map<NT> & map,
#endif
                                                                   const std::string & lbl)
{
   const RCP<Thyra::VectorBase<ST> > thyraVec  // need a Thyra::VectorBase object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
         = Thyra::createVector<ST,LO,GO,NT>(tv,Thyra::createVectorSpace<ST,LO,GO,NT>(rcpFromRef(map)));
#else
         = Thyra::createVector<ST,NT>(tv,Thyra::createVectorSpace<ST,NT>(rcpFromRef(map)));
#endif
   Teuchos::RCP<Thyra::LinearOpBase<ST> > op 
         = Teuchos::rcp(new Thyra::DefaultDiagonalLinearOp<ST>(thyraVec));
   op->setObjectLabel(lbl);
   return op;
}

/** \brief Fill a Thyra vector with the contents of an epetra vector. This prevents the
  *
  * Fill a Thyra vector with the contents of an epetra vector. This prevents the need
  * to reallocate memory using a create_MultiVector routine. It also allows an aritrary
  * Thyra vector to be filled.
  *
  * \param[in,out] spmdMV Multi-vector to be filled.
  * \param[in]     mv     Epetra multi-vector to be used in filling the Thyra vector.
  */    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
void fillDefaultSpmdMultiVector(Teuchos::RCP<Thyra::TpetraMultiVector<ST,LO,GO,NT> > & spmdMV,
                                Teuchos::RCP<Tpetra::MultiVector<ST,LO,GO,NT> > & tpetraMV)
#else
void fillDefaultSpmdMultiVector(Teuchos::RCP<Thyra::TpetraMultiVector<ST,NT> > & spmdMV,
                                Teuchos::RCP<Tpetra::MultiVector<ST,NT> > & tpetraMV)
#endif
{
   // first get desired range and domain
   //const RCP<const Thyra::SpmdVectorSpaceBase<ST> > range  = spmdMV->spmdSpace();
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
   const RCP<Thyra::TpetraVectorSpace<ST,LO,GO,NT> > range  = Thyra::tpetraVectorSpace<ST,LO,GO,NT>(tpetraMV->getMap());
#else
   const RCP<Thyra::TpetraVectorSpace<ST,NT> > range  = Thyra::tpetraVectorSpace<ST,NT>(tpetraMV->getMap());
#endif
   const RCP<const Thyra::ScalarProdVectorSpaceBase<ST> > domain 
         = rcp_dynamic_cast<const Thyra::ScalarProdVectorSpaceBase<ST> >(spmdMV->domain());

   TEUCHOS_ASSERT((size_t) domain->dim()==tpetraMV->getNumVectors());

   // New local view of raw data
   if(!tpetraMV->isConstantStride()) 
      TEUCHOS_TEST_FOR_EXCEPT(true); // ToDo: Implement views of non-contiguous mult-vectors!

   // Build the MultiVector
   spmdMV->initialize(range, domain, tpetraMV);

   // make sure the Epetra_MultiVector doesn't disappear prematurely
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
   Teuchos::set_extra_data<RCP<Tpetra::MultiVector<ST,LO,GO,NT> > >(tpetraMV,"Tpetra::MultiVector",Teuchos::outArg(spmdMV));
#else
   Teuchos::set_extra_data<RCP<Tpetra::MultiVector<ST,NT> > >(tpetraMV,"Tpetra::MultiVector",Teuchos::outArg(spmdMV));
#endif
}

/** \brief Build a vector of the dirchlet row indices. 
  *
  * Build a vector of the dirchlet row indices. That is, record the global
  * index of any row that is all zeros except for $1$ on the diagonal.
  *
  * \param[in]     rowMap   Map specifying which global indices this process examines 
  * \param[in]     mat      Matrix to be examined
  * \param[in,out] indices Output list of indices corresponding to dirchlet rows (GIDs).
  */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
void identityRowIndices(const Tpetra::Map<LO,GO,NT> & rowMap, const Tpetra::CrsMatrix<ST,LO,GO,NT> & mat,std::vector<GO> & outIndices)
#else
void identityRowIndices(const Tpetra::Map<NT> & rowMap, const Tpetra::CrsMatrix<ST,NT> & mat,std::vector<GO> & outIndices)
#endif
{
   GO maxSz = mat.getGlobalMaxNumRowEntries();
   std::vector<ST> values(maxSz);
   std::vector<GO> indices(maxSz);

   // loop over elements owned by this processor
   for(size_t i=0;i<rowMap.getNodeNumElements();i++) {
      bool rowIsIdentity = true;
      GO rowGID = rowMap.getGlobalElement(i);

      size_t numEntries = mat.getNumEntriesInGlobalRow (i);
      std::vector<GO> indices(numEntries);
      std::vector<ST> values(numEntries);
      const Teuchos::ArrayView<GO> indices_av(indices);
      const Teuchos::ArrayView<ST> values_av(values);

      mat.getGlobalRowCopy(rowGID,indices_av,values_av,numEntries);

      // loop over the columns of this row
      for(size_t j=0;j<numEntries;j++) {
         GO colGID = indices_av[j];

         // look at row entries
         if(colGID==rowGID) rowIsIdentity &= values_av[j]==1.0;
         else               rowIsIdentity &= values_av[j]==0.0;

         // not a dirchlet row...quit
         if(not rowIsIdentity)
            break;
      }

      // save a row that is dirchlet
      if(rowIsIdentity)
         outIndices.push_back(rowGID);
   }
}

/** \brief Zero out the value of a vector on the specified
  *        set of global indices.
  *
  * Zero out the value of a vector on the specified set of global
  * indices. The indices here are assumed to belong to the calling
  * process (i.e. zeroIndices $\in$ mv.Map()).
  *
  * \param[in,out] mv           Vector whose entries will be zeroed
  * \param[in]     zeroIndices Indices local to this process that need to be zeroed
  */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
void zeroMultiVectorRowIndices(Tpetra::MultiVector<ST,LO,GO,NT> & mv,const std::vector<GO> & zeroIndices)
#else
void zeroMultiVectorRowIndices(Tpetra::MultiVector<ST,NT> & mv,const std::vector<GO> & zeroIndices)
#endif
{
   LO colCnt = mv.getNumVectors();
   std::vector<GO>::const_iterator itr;
 
   // loop over the indices to zero
   for(itr=zeroIndices.begin();itr!=zeroIndices.end();++itr) {
 
      // loop over columns
      for(int j=0;j<colCnt;j++)
         mv.replaceGlobalValue(*itr,j,0.0);
   }
}

/** \brief Constructor for a ZeroedOperator.
  *
  * Build a ZeroedOperator based on a particular Epetra_Operator and
  * a set of indices to zero out. These indices must be local to this
  * processor as specified by RowMap().
  *
  * \param[in] zeroIndices Set of indices to zero out (must be local).
  * \param[in] op           Underlying epetra operator to use.
  */
ZeroedOperator::ZeroedOperator(const std::vector<GO> & zeroIndices,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                               const Teuchos::RCP<const Tpetra::Operator<ST,LO,GO,NT> > & op)
#else
                               const Teuchos::RCP<const Tpetra::Operator<ST,NT> > & op)
#endif
   : zeroIndices_(zeroIndices), tpetraOp_(op)
{ }

//! Perform a matrix-vector product with certain rows zeroed out
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
void ZeroedOperator::apply(const Tpetra::MultiVector<ST,LO,GO,NT> & X, Tpetra::MultiVector<ST,LO,GO,NT> & Y, Teuchos::ETransp mode, ST alpha, ST beta) const
#else
void ZeroedOperator::apply(const Tpetra::MultiVector<ST,NT> & X, Tpetra::MultiVector<ST,NT> & Y, Teuchos::ETransp mode, ST alpha, ST beta) const
#endif
{
/*
   Epetra_MultiVector temp(X);
   zeroMultiVectorRowIndices(temp,zeroIndices_);
   int result = epetraOp_->Apply(temp,Y);
*/

   tpetraOp_->apply(X,Y,mode,alpha,beta);

   // zero a few of the rows
   zeroMultiVectorRowIndices(Y,zeroIndices_);
}

bool isTpetraLinearOp(const LinearOp & op)
{
   // See if the operator is a TpetraLinearOp
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
   RCP<const Thyra::TpetraLinearOp<ST,LO,GO,NT> > tOp = rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST,LO,GO,NT> >(op);
#else
   RCP<const Thyra::TpetraLinearOp<ST,NT> > tOp = rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST,NT> >(op);
#endif
   if (!tOp.is_null())
     return true;

   // See if the operator is a wrapped TpetraLinearOp
   ST scalar = 0.0;
   Thyra::EOpTransp transp = Thyra::NOTRANS;
   RCP<const Thyra::LinearOpBase<ST> > wrapped_op;
   Thyra::unwrap(op, &scalar, &transp, &wrapped_op);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
   tOp = rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST,LO,GO,NT> >(wrapped_op);
#else
   tOp = rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST,NT> >(wrapped_op);
#endif
   if (!tOp.is_null())
     return true;

   return false;
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
RCP<const Tpetra::CrsMatrix<ST,LO,GO,NT> > getTpetraCrsMatrix(const LinearOp & op, ST *scalar, bool *transp)
#else
RCP<const Tpetra::CrsMatrix<ST,NT> > getTpetraCrsMatrix(const LinearOp & op, ST *scalar, bool *transp)
#endif
{
    // If the operator is a TpetraLinearOp
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<const Thyra::TpetraLinearOp<ST,LO,GO,NT> > tOp = rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST,LO,GO,NT> >(op);
#else
    RCP<const Thyra::TpetraLinearOp<ST,NT> > tOp = rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST,NT> >(op);
#endif
    if(!tOp.is_null()){
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<const Tpetra::CrsMatrix<ST,LO,GO,NT> > matrix = rcp_dynamic_cast<const Tpetra::CrsMatrix<ST,LO,GO,NT> >(tOp->getConstTpetraOperator(),true);
#else
      RCP<const Tpetra::CrsMatrix<ST,NT> > matrix = rcp_dynamic_cast<const Tpetra::CrsMatrix<ST,NT> >(tOp->getConstTpetraOperator(),true);
#endif
      *scalar = 1.0;
      *transp = false;
      return matrix;
    }

    // If the operator is a wrapped TpetraLinearOp
    RCP<const Thyra::LinearOpBase<ST> > wrapped_op;
    Thyra::EOpTransp eTransp = Thyra::NOTRANS;
    Thyra::unwrap(op, scalar, &eTransp, &wrapped_op);
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    tOp = rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST,LO,GO,NT> >(wrapped_op,true);
#else
    tOp = rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST,NT> >(wrapped_op,true);
#endif
    if(!tOp.is_null()){
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      RCP<const Tpetra::CrsMatrix<ST,LO,GO,NT> > matrix = rcp_dynamic_cast<const Tpetra::CrsMatrix<ST,LO,GO,NT> >(tOp->getConstTpetraOperator(),true);
#else
      RCP<const Tpetra::CrsMatrix<ST,NT> > matrix = rcp_dynamic_cast<const Tpetra::CrsMatrix<ST,NT> >(tOp->getConstTpetraOperator(),true);
#endif
      *transp = true;
      if(eTransp == Thyra::NOTRANS)
        *transp = false;
      return matrix;
    }

    return Teuchos::null;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
RCP<const Tpetra::CrsMatrix<ST,LO,GO,NT> > epetraCrsMatrixToTpetra(const RCP<const Epetra_CrsMatrix> A_e, const RCP<const Teuchos::Comm<int> > comm)
#else
RCP<const Tpetra::CrsMatrix<ST,NT> > epetraCrsMatrixToTpetra(const RCP<const Epetra_CrsMatrix> A_e, const RCP<const Teuchos::Comm<int> > comm)
#endif
{
   int* ptr;
   int* ind;
   double* val;
 
   int info = A_e->ExtractCrsDataPointers (ptr, ind, val);
   TEUCHOS_TEST_FOR_EXCEPTION(info!=0,std::logic_error, "Could not extract data from Epetra_CrsMatrix");
   const LO numRows = A_e->Graph ().NumMyRows ();
   const LO nnz = A_e->Graph ().NumMyEntries ();

   Teuchos::ArrayRCP<size_t> ptr2 (numRows+1);
   Teuchos::ArrayRCP<int> ind2 (nnz);
   Teuchos::ArrayRCP<double> val2 (nnz);

   std::copy (ptr, ptr + numRows + 1, ptr2.begin ());
   std::copy (ind, ind + nnz, ind2.begin ());
   std::copy (val, val + nnz, val2.begin ());

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
   RCP<const Tpetra::Map<LO,GO,NT> > rowMap = epetraMapToTpetra(A_e->RowMap(),comm);
   RCP<Tpetra::CrsMatrix<ST,LO,GO,NT> > A_t = Tpetra::createCrsMatrix<ST,LO,GO,NT>(rowMap, A_e->GlobalMaxNumEntries());
#else
   RCP<const Tpetra::Map<NT> > rowMap = epetraMapToTpetra(A_e->RowMap(),comm);
   RCP<Tpetra::CrsMatrix<ST,NT> > A_t = Tpetra::createCrsMatrix<ST,NT>(rowMap, A_e->GlobalMaxNumEntries());
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
   RCP<const Tpetra::Map<LO,GO,NT> > domainMap = epetraMapToTpetra(A_e->OperatorDomainMap(),comm);
   RCP<const Tpetra::Map<LO,GO,NT> > rangeMap = epetraMapToTpetra(A_e->OperatorRangeMap(),comm);
   RCP<const Tpetra::Map<LO,GO,NT> > colMap = epetraMapToTpetra(A_e->ColMap(),comm);
#else
   RCP<const Tpetra::Map<NT> > domainMap = epetraMapToTpetra(A_e->OperatorDomainMap(),comm);
   RCP<const Tpetra::Map<NT> > rangeMap = epetraMapToTpetra(A_e->OperatorRangeMap(),comm);
   RCP<const Tpetra::Map<NT> > colMap = epetraMapToTpetra(A_e->ColMap(),comm);
#endif

   A_t->replaceColMap(colMap);
   A_t->setAllValues (ptr2, ind2, val2);
   A_t->fillComplete(domainMap,rangeMap);
   return A_t;

}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
RCP<Tpetra::CrsMatrix<ST,LO,GO,NT> > nonConstEpetraCrsMatrixToTpetra(const RCP<Epetra_CrsMatrix> A_e, const RCP<const Teuchos::Comm<int> > comm)
#else
RCP<Tpetra::CrsMatrix<ST,NT> > nonConstEpetraCrsMatrixToTpetra(const RCP<Epetra_CrsMatrix> A_e, const RCP<const Teuchos::Comm<int> > comm)
#endif
{
   int* ptr;
   int* ind;
   double* val;
 
   int info = A_e->ExtractCrsDataPointers (ptr, ind, val);
   TEUCHOS_TEST_FOR_EXCEPTION(info!=0,std::logic_error, "Could not extract data from Epetra_CrsMatrix");
   const LO numRows = A_e->Graph ().NumMyRows ();
   const LO nnz = A_e->Graph ().NumMyEntries ();

   Teuchos::ArrayRCP<size_t> ptr2 (numRows+1);
   Teuchos::ArrayRCP<int> ind2 (nnz);
   Teuchos::ArrayRCP<double> val2 (nnz);

   std::copy (ptr, ptr + numRows + 1, ptr2.begin ());
   std::copy (ind, ind + nnz, ind2.begin ());
   std::copy (val, val + nnz, val2.begin ());

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
   RCP<const Tpetra::Map<LO,GO,NT> > rowMap = epetraMapToTpetra(A_e->RowMap(),comm);
   RCP<Tpetra::CrsMatrix<ST,LO,GO,NT> > A_t = Tpetra::createCrsMatrix<ST,LO,GO,NT>(rowMap, A_e->GlobalMaxNumEntries());
#else
   RCP<const Tpetra::Map<NT> > rowMap = epetraMapToTpetra(A_e->RowMap(),comm);
   RCP<Tpetra::CrsMatrix<ST,NT> > A_t = Tpetra::createCrsMatrix<ST,NT>(rowMap, A_e->GlobalMaxNumEntries());
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
   RCP<const Tpetra::Map<LO,GO,NT> > domainMap = epetraMapToTpetra(A_e->OperatorDomainMap(),comm);
   RCP<const Tpetra::Map<LO,GO,NT> > rangeMap = epetraMapToTpetra(A_e->OperatorRangeMap(),comm);
   RCP<const Tpetra::Map<LO,GO,NT> > colMap = epetraMapToTpetra(A_e->ColMap(),comm);
#else
   RCP<const Tpetra::Map<NT> > domainMap = epetraMapToTpetra(A_e->OperatorDomainMap(),comm);
   RCP<const Tpetra::Map<NT> > rangeMap = epetraMapToTpetra(A_e->OperatorRangeMap(),comm);
   RCP<const Tpetra::Map<NT> > colMap = epetraMapToTpetra(A_e->ColMap(),comm);
#endif

   A_t->replaceColMap(colMap);
   A_t->setAllValues (ptr2, ind2, val2);
   A_t->fillComplete(domainMap,rangeMap);
   return A_t;

}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
RCP<const Tpetra::Map<LO,GO,NT> > epetraMapToTpetra(const Epetra_Map eMap, const RCP<const Teuchos::Comm<int> > comm)
#else
RCP<const Tpetra::Map<NT> > epetraMapToTpetra(const Epetra_Map eMap, const RCP<const Teuchos::Comm<int> > comm)
#endif
{
  std::vector<int> intGIDs(eMap.NumMyElements());
  eMap.MyGlobalElements(&intGIDs[0]);

  std::vector<GO> myGIDs(eMap.NumMyElements());
  for(int k = 0; k < eMap.NumMyElements(); k++)
    myGIDs[k] = (GO) intGIDs[k];

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  return rcp(new const Tpetra::Map<LO,GO,NT>(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),Teuchos::ArrayView<GO>(myGIDs),0,comm));
#else
  return rcp(new const Tpetra::Map<NT>(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),Teuchos::ArrayView<GO>(myGIDs),0,comm));
#endif
}

} // end namespace TpetraHelpers
} // end namespace Teko
