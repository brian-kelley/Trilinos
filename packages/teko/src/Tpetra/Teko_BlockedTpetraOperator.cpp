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

#include "Teko_BlockedTpetraOperator.hpp"
#include "Teko_TpetraBlockedMappingStrategy.hpp"
#include "Teko_TpetraReorderedMappingStrategy.hpp"

#include "Teuchos_VerboseObject.hpp"

#include "Thyra_LinearOpBase.hpp"
#include "Thyra_TpetraLinearOp.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"
#include "Thyra_DefaultProductMultiVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"
#include "Thyra_DefaultBlockedLinearOp.hpp"

#include "MatrixMarket_Tpetra.hpp"

#include "Teko_Utilities.hpp"

namespace Teko {
namespace TpetraHelpers {

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_dynamic_cast;

BlockedTpetraOperator::BlockedTpetraOperator(const std::vector<std::vector<GO> > & vars,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                                             const Teuchos::RCP<const Tpetra::Operator<ST,LO,GO,NT> > & content,
#else
                                             const Teuchos::RCP<const Tpetra::Operator<ST,NT> > & content,
#endif
                                             const std::string & label) 
      : Teko::TpetraHelpers::TpetraOperatorWrapper(), label_(label)
{
   SetContent(vars,content);
}

void BlockedTpetraOperator::SetContent(const std::vector<std::vector<GO> > & vars,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                                       const Teuchos::RCP<const Tpetra::Operator<ST,LO,GO,NT> > & content)
#else
                                       const Teuchos::RCP<const Tpetra::Operator<ST,NT> > & content)
#endif
{ 
   fullContent_ = content;
   blockedMapping_ = rcp(new TpetraBlockedMappingStrategy(vars,fullContent_->getDomainMap(),
                                                         *fullContent_->getDomainMap()->getComm()));
   SetMapStrategy(blockedMapping_);

   // build thyra operator
   BuildBlockedOperator(); 
}

void BlockedTpetraOperator::BuildBlockedOperator()
{
   TEUCHOS_ASSERT(blockedMapping_!=Teuchos::null);

   // get a CRS matrix
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
   const RCP<const Tpetra::CrsMatrix<ST,LO,GO,NT> > crsContent 
         = rcp_dynamic_cast<const Tpetra::CrsMatrix<ST,LO,GO,NT> >(fullContent_);
#else
   const RCP<const Tpetra::CrsMatrix<ST,NT> > crsContent 
         = rcp_dynamic_cast<const Tpetra::CrsMatrix<ST,NT> >(fullContent_);
#endif

   // ask the strategy to build the Thyra operator for you
   if(blockedOperator_==Teuchos::null) {
      blockedOperator_ = blockedMapping_->buildBlockedThyraOp(crsContent,label_);
   }
   else {
      const RCP<Thyra::BlockedLinearOpBase<ST> > blkOp 
            = rcp_dynamic_cast<Thyra::BlockedLinearOpBase<ST> >(blockedOperator_,true);
      blockedMapping_->rebuildBlockedThyraOp(crsContent,blkOp);
   }

   // set whatever is returned
   SetOperator(blockedOperator_,false);

   // reorder if neccessary
   if(reorderManager_!=Teuchos::null) 
      Reorder(*reorderManager_);
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
const Teuchos::RCP<const Tpetra::Operator<ST,LO,GO,NT> > BlockedTpetraOperator::GetBlock(int i,int j) const
#else
const Teuchos::RCP<const Tpetra::Operator<ST,NT> > BlockedTpetraOperator::GetBlock(int i,int j) const
#endif
{
   const RCP<const Thyra::BlockedLinearOpBase<ST> > blkOp 
         = Teuchos::rcp_dynamic_cast<const Thyra::BlockedLinearOpBase<ST> >(getThyraOp());

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
   RCP<const Thyra::TpetraLinearOp<ST,LO,GO,NT> > tOp = rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST,LO,GO,NT> >(blkOp->getBlock(i,j),true);
#else
   RCP<const Thyra::TpetraLinearOp<ST,NT> > tOp = rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST,NT> >(blkOp->getBlock(i,j),true);
#endif
   return tOp->getConstTpetraOperator();
}

/** Use a reorder manager to block this operator as desired.
  * Multiple calls to the function reorder only the underlying object. 
  */
void BlockedTpetraOperator::Reorder(const BlockReorderManager & brm)
{
   reorderManager_ = rcp(new BlockReorderManager(brm));

   // build reordered objects
   RCP<const MappingStrategy> reorderMapping = rcp(new TpetraReorderedMappingStrategy(*reorderManager_,blockedMapping_));
   RCP<const Thyra::BlockedLinearOpBase<ST> > blockOp
         = rcp_dynamic_cast<const Thyra::BlockedLinearOpBase<ST> >(blockedOperator_);

   RCP<const Thyra::LinearOpBase<ST> > A = buildReorderedLinearOp(*reorderManager_,blockOp);

   // set them as working values
   SetMapStrategy(reorderMapping);
   SetOperator(A,false);
}

//! Remove any reordering on this object
void BlockedTpetraOperator::RemoveReording()
{
   SetMapStrategy(blockedMapping_);
   SetOperator(blockedOperator_,false);
   reorderManager_ = Teuchos::null;
}

/** Write out this operator to matrix market files
  */
void BlockedTpetraOperator::WriteBlocks(const std::string & prefix) const
{
   RCP<Thyra::PhysicallyBlockedLinearOpBase<ST> > blockOp
         = rcp_dynamic_cast<Thyra::PhysicallyBlockedLinearOpBase<ST> >(blockedOperator_);

   // get size of blocked block operator
   int rows = Teko::blockRowCount(blockOp);

   for(int i=0;i<rows;i++) {
      for(int j=0;j<rows;j++) {
         // build the file name
         std::stringstream ss;
         ss << prefix << "_" << i << j << ".mm";

         // get the row matrix object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
         RCP<const Thyra::TpetraLinearOp<ST,LO,GO,NT> > tOp = rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST,LO,GO,NT> >(blockOp->getBlock(i,j));
         RCP<const Tpetra::CrsMatrix<ST,LO,GO,NT> > mat
               = Teuchos::rcp_dynamic_cast<const Tpetra::CrsMatrix<ST,LO,GO,NT> >(tOp->getConstTpetraOperator());
#else
         RCP<const Thyra::TpetraLinearOp<ST,NT> > tOp = rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST,NT> >(blockOp->getBlock(i,j));
         RCP<const Tpetra::CrsMatrix<ST,NT> > mat
               = Teuchos::rcp_dynamic_cast<const Tpetra::CrsMatrix<ST,NT> >(tOp->getConstTpetraOperator());
#endif

         // write to file
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
         Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<ST,LO,GO,NT> >::writeSparseFile(ss.str().c_str(),mat);
#else
         Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<ST,NT> >::writeSparseFile(ss.str().c_str(),mat);
#endif
      }
   }
}

bool BlockedTpetraOperator::testAgainstFullOperator(int count,ST tol) const
{
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
   Tpetra::Vector<ST,LO,GO,NT> xf(getRangeMap());
   Tpetra::Vector<ST,LO,GO,NT> xs(getRangeMap());
   Tpetra::Vector<ST,LO,GO,NT> y(getDomainMap());
#else
   Tpetra::Vector<ST,NT> xf(getRangeMap());
   Tpetra::Vector<ST,NT> xs(getRangeMap());
   Tpetra::Vector<ST,NT> y(getDomainMap());
#endif

   // test operator many times
   bool result = true;
   ST diffNorm=0.0,trueNorm=0.0;
   for(int i=0;i<count;i++) {
      xf.putScalar(0.0);
      xs.putScalar(0.0);
      y.randomize();

      // apply operator
      apply(y,xs); // xs = A*y
      fullContent_->apply(y,xf); // xf = A*y

      // compute norms
      xs.update(-1.0,xf,1.0);
      diffNorm = xs.norm2();
      trueNorm = xf.norm2();

      // check result
      result &= (diffNorm/trueNorm < tol);
   }

   return result;
}

} // end namespace TpetraHelpers
} // end namespace Teko
