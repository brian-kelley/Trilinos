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

#ifndef _FROSCH_PARTITIONOFUNITY_DEF_HPP
#define _FROSCH_PARTITIONOFUNITY_DEF_HPP

#include <FROSch_PartitionOfUnity_decl.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    PartitionOfUnity<SC,LO,GO,NO>::PartitionOfUnity(CommPtr mpiComm,
#else
    template <class SC,class NO>
    PartitionOfUnity<SC,NO>::PartitionOfUnity(CommPtr mpiComm,
#endif
                                                    CommPtr serialComm,
                                                    UN dofsPerNode,
                                                    ConstXMapPtr nodesMap,
                                                    ConstXMapPtrVecPtr dofsMaps,
                                                    ParameterListPtr parameterList,
                                                    Verbosity verbosity,
                                                    UN levelID) :
    MpiComm_ (mpiComm),
    SerialComm_ (serialComm),
    ParameterList_ (parameterList),
    Verbose_ (MpiComm_->getRank() == 0),
    Verbosity_ (verbosity),
    LevelID_ (levelID)
    {
        
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    PartitionOfUnity<SC,LO,GO,NO>::~PartitionOfUnity()
#else
    template <class SC,class NO>
    PartitionOfUnity<SC,NO>::~PartitionOfUnity()
#endif
    {

    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int PartitionOfUnity<SC,LO,GO,NO>::assembledPartitionOfUnityMaps()
#else
    template <class SC,class NO>
    int PartitionOfUnity<SC,NO>::assembledPartitionOfUnityMaps()
#endif
    {
        if (!AssmbledPartitionOfUnityMap_.is_null()) {
            FROSCH_NOTIFICATION("FROSch::PartitionOfUnity",Verbosity_,"AssmbledPartitionOfUnityMap_ has already been assembled previously.");
        }
        LOVecPtr2D partMappings;
        AssmbledPartitionOfUnityMap_ = AssembleMaps(PartitionOfUnityMaps_(),partMappings);
        return 0;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    typename PartitionOfUnity<SC,LO,GO,NO>::XMultiVectorPtrVecPtr PartitionOfUnity<SC,LO,GO,NO>::getLocalPartitionOfUnity() const
#else
    template <class SC,class NO>
    typename PartitionOfUnity<SC,NO>::XMultiVectorPtrVecPtr PartitionOfUnity<SC,NO>::getLocalPartitionOfUnity() const
#endif
    {
        return LocalPartitionOfUnity_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    typename PartitionOfUnity<SC,LO,GO,NO>::XMapPtrVecPtr PartitionOfUnity<SC,LO,GO,NO>::getPartitionOfUnityMaps() const
#else
    template <class SC,class NO>
    typename PartitionOfUnity<SC,NO>::XMapPtrVecPtr PartitionOfUnity<SC,NO>::getPartitionOfUnityMaps() const
#endif
    {
        return PartitionOfUnityMaps_;
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    typename PartitionOfUnity<SC,LO,GO,NO>::XMapPtr PartitionOfUnity<SC,LO,GO,NO>::getAssembledPartitionOfUnityMap() const
#else
    template <class SC,class NO>
    typename PartitionOfUnity<SC,NO>::XMapPtr PartitionOfUnity<SC,NO>::getAssembledPartitionOfUnityMap() const
#endif
    {
        return AssmbledPartitionOfUnityMap_;
    }
}

#endif
