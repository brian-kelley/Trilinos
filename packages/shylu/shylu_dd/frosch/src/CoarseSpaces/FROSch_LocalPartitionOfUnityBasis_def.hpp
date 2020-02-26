//@HEADER
// ************************************************************************
//
//               ShyLU: Hybrid preconditioner package
//                 Copyright 2012 Sandia Corporation
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
// Questions? Contact Alexander Heinlein (alexander.heinlein@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#ifndef _FROSCH_PARTITIONOFUNITYBASIS_DEF_hpp
#define _FROSCH_PARTITIONOFUNITYBASIS_DEF_hpp

#include <FROSch_LocalPartitionOfUnityBasis_decl.hpp>


namespace FROSch {
    
    using namespace Teuchos;
    using namespace Xpetra;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    LocalPartitionOfUnityBasis<SC,LO,GO,NO>::LocalPartitionOfUnityBasis(CommPtr mpiComm,
#else
    template<class SC,class NO>
    LocalPartitionOfUnityBasis<SC,NO>::LocalPartitionOfUnityBasis(CommPtr mpiComm,
#endif
                                                                        CommPtr serialComm,
                                                                        UN dofsPerNode,
                                                                        ParameterListPtr parameterList,
                                                                        ConstXMultiVectorPtr nullSpaceBasis,
                                                                        XMultiVectorPtrVecPtr partitionOfUnity,
                                                                        XMapPtrVecPtr partitionOfUnityMaps) :
    MpiComm_ (mpiComm),
    SerialComm_ (serialComm),
    DofsPerNode_ (dofsPerNode),
    ParameterList_ (parameterList),
    PartitionOfUnity_ (partitionOfUnity),
    NullspaceBasis_ (nullSpaceBasis),
    PartitionOfUnityMaps_ (partitionOfUnityMaps)
    {

    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int LocalPartitionOfUnityBasis<SC,LO,GO,NO>::addPartitionOfUnity(XMultiVectorPtrVecPtr partitionOfUnity,
#else
    template<class SC,class NO>
    int LocalPartitionOfUnityBasis<SC,NO>::addPartitionOfUnity(XMultiVectorPtrVecPtr partitionOfUnity,
#endif
                                                                     XMapPtrVecPtr partitionOfUnityMaps)
    {
        PartitionOfUnity_ = partitionOfUnity;
        PartitionOfUnityMaps_ = partitionOfUnityMaps;
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int LocalPartitionOfUnityBasis<SC,LO,GO,NO>::addGlobalBasis(ConstXMultiVectorPtr nullSpaceBasis)
#else
    template<class SC,class NO>
    int LocalPartitionOfUnityBasis<SC,NO>::addGlobalBasis(ConstXMultiVectorPtr nullSpaceBasis)
#endif
    {
        NullspaceBasis_ = nullSpaceBasis;
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    int LocalPartitionOfUnityBasis<SC,LO,GO,NO>::buildLocalPartitionOfUnityBasis()
#else
    template<class SC,class NO>
    int LocalPartitionOfUnityBasis<SC,NO>::buildLocalPartitionOfUnityBasis()
#endif
    {
        FROSCH_ASSERT(!NullspaceBasis_.is_null(),"Nullspace Basis is not set.");
        FROSCH_ASSERT(!PartitionOfUnity_.is_null(),"Partition Of Unity is not set.");
        FROSCH_ASSERT(!PartitionOfUnityMaps_.is_null(),"Partition Of Unity Map is not set.");

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        LocalPartitionOfUnitySpace_ = CoarseSpacePtr(new CoarseSpace<SC,LO,GO,NO>(this->MpiComm_,this->SerialComm_));
#else
        LocalPartitionOfUnitySpace_ = CoarseSpacePtr(new CoarseSpace<SC,NO>(this->MpiComm_,this->SerialComm_));
#endif

        XMultiVectorPtrVecPtr2D tmpBasis(PartitionOfUnity_.size());
        ConstXMapPtr nullspaceBasisMap = NullspaceBasis_->getMap();
        for (UN i=0; i<PartitionOfUnity_.size(); i++) {
            if (!PartitionOfUnity_[i].is_null()) {
                FROSCH_ASSERT(PartitionOfUnityMaps_[i]->getNodeNumElements()>0,"PartitionOfUnityMaps_[i]->getNodeNumElements()==0");
                tmpBasis[i] = XMultiVectorPtrVecPtr(PartitionOfUnity_[i]->getNumVectors());
                for (UN j=0; j<PartitionOfUnity_[i]->getNumVectors(); j++) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                    XMultiVectorPtr tmpBasisJ = MultiVectorFactory<SC,LO,GO,NO>::Build(nullspaceBasisMap,NullspaceBasis_->getNumVectors());
#else
                    XMultiVectorPtr tmpBasisJ = MultiVectorFactory<SC,NO>::Build(nullspaceBasisMap,NullspaceBasis_->getNumVectors());
#endif
                    tmpBasisJ->elementWiseMultiply(ScalarTraits<SC>::one(),*PartitionOfUnity_[i]->getVector(j),*NullspaceBasis_,ScalarTraits<SC>::one());
                    if (ParameterList_->get("Orthogonalize",true)) {
                        tmpBasis[i][j] = ModifiedGramSchmidt(tmpBasisJ.getConst());
                    } else {
                        tmpBasis[i][j] = tmpBasisJ;
                    }
                }
            } else {
                FROSCH_ASSERT(PartitionOfUnityMaps_[i]->getNodeNumElements()==0,"PartitionOfUnityMaps_[i]->getNodeNumElements()!=0");
            }
        }

        // Determine Number of Basisfunctions per Entity
        UNVecPtr maxNV(PartitionOfUnity_.size());
        for (UN i=0; i<PartitionOfUnity_.size(); i++) {
            UN maxNVLocal = 0;
            if (!PartitionOfUnityMaps_[i].is_null()) {
                if (!PartitionOfUnity_[i].is_null()) {
                    for (UN j=0; j<tmpBasis[i].size(); j++) {
                        maxNVLocal = std::max(maxNVLocal,(UN) tmpBasis[i][j]->getNumVectors());
                    }
                }
                reduceAll(*MpiComm_,REDUCE_MAX,maxNVLocal,ptr(&maxNV[i]));
            }
        }

        if (!ParameterList_->get("Number of Basisfunctions per Entity","MaxAll").compare("MaxAll")) {
            UNVecPtr::iterator max = std::max_element(maxNV.begin(),maxNV.end());
            for (UN i=0; i<maxNV.size(); i++) {
                maxNV[i] = *max;
            }
        } else if (!ParameterList_->get("Number of Basisfunctions per Entity","MaxAll").compare("MaxEntityType")) {

        } else {
            FROSCH_ASSERT(false,"Number of Basisfunctions per Entity type is unknown.");
        } // Testen!!!!!!!!!!!!!!!!!!!!!!!! AUSGABE IMPLEMENTIEREN!!!!!!

        // Kann man das schöner machen?
        for (UN i=0; i<PartitionOfUnity_.size(); i++) {
            if (!PartitionOfUnityMaps_[i].is_null()) {
                if (!PartitionOfUnity_[i].is_null()) {
                    ConstXMapPtr partitionOfUnityMap_i = PartitionOfUnity_[i]->getMap();
                    for (UN j=0; j<maxNV[i]; j++) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                        XMultiVectorPtr entityBasis = MultiVectorFactory<SC,LO,GO,NO >::Build(partitionOfUnityMap_i,PartitionOfUnity_[i]->getNumVectors());
#else
                        XMultiVectorPtr entityBasis = MultiVectorFactory<SC,NO >::Build(partitionOfUnityMap_i,PartitionOfUnity_[i]->getNumVectors());
#endif
                        entityBasis->scale(ScalarTraits<SC>::zero());
                        for (UN k=0; k<PartitionOfUnity_[i]->getNumVectors(); k++) {
                            if (j<tmpBasis[i][k]->getNumVectors()) {
                                entityBasis->getDataNonConst(k).deepCopy(tmpBasis[i][k]->getData(j)()); // Here, we copy data. Do we need to do this?
                            }
                        }
                        LocalPartitionOfUnitySpace_->addSubspace(PartitionOfUnityMaps_[i],null,entityBasis);
                    }
                } else {
                    for (UN j=0; j<maxNV[i]; j++) {
                        LocalPartitionOfUnitySpace_->addSubspace(PartitionOfUnityMaps_[i]);
                    }
                }
            } else {
                FROSCH_WARNING("FROSch::LocalPartitionOfUnityBasis",this->MpiComm_->getRank()==0,"PartitionOfUnityMaps_[i].is_null()");
            }
        }

        LocalPartitionOfUnitySpace_->assembleCoarseSpace();

        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    typename LocalPartitionOfUnityBasis<SC,LO,GO,NO>::XMultiVectorPtrVecPtr LocalPartitionOfUnityBasis<SC,LO,GO,NO>::getPartitionOfUnity() const
#else
    template<class SC,class NO>
    typename LocalPartitionOfUnityBasis<SC,NO>::XMultiVectorPtrVecPtr LocalPartitionOfUnityBasis<SC,NO>::getPartitionOfUnity() const
#endif
    {
        FROSCH_ASSERT(!PartitionOfUnity_.is_null(),"Partition Of Unity is not set.");
        return PartitionOfUnity_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    typename LocalPartitionOfUnityBasis<SC,LO,GO,NO>::XMultiVectorPtr LocalPartitionOfUnityBasis<SC,LO,GO,NO>::getNullspaceBasis() const
#else
    template<class SC,class NO>
    typename LocalPartitionOfUnityBasis<SC,NO>::XMultiVectorPtr LocalPartitionOfUnityBasis<SC,NO>::getNullspaceBasis() const
#endif
    {
        FROSCH_ASSERT(!NullspaceBasis_.is_null(),"Nullspace Basis is not set.");
        return NullspaceBasis_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class SC,class LO,class GO,class NO>
    typename LocalPartitionOfUnityBasis<SC,LO,GO,NO>::CoarseSpacePtr LocalPartitionOfUnityBasis<SC,LO,GO,NO>::getLocalPartitionOfUnitySpace() const
#else
    template<class SC,class NO>
    typename LocalPartitionOfUnityBasis<SC,NO>::CoarseSpacePtr LocalPartitionOfUnityBasis<SC,NO>::getLocalPartitionOfUnitySpace() const
#endif
    {
        FROSCH_ASSERT(!LocalPartitionOfUnitySpace_.is_null(),"Local Partition Of Unity Space is not built yet.");
        return LocalPartitionOfUnitySpace_;
    }
}

#endif
