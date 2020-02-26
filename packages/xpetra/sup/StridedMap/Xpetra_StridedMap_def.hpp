// @HEADER
//
// ***********************************************************************
//
//             Xpetra: A linear algebra interface package
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

// WARNING: This code is experimental. Backwards compatibility should not be expected.

#ifndef XPETRA_STRIDEDMAP_DEF_HPP
#define XPETRA_STRIDEDMAP_DEF_HPP

#include "Xpetra_StridedMap.hpp"

#include <Teuchos_OrdinalTraits.hpp>

#include "Xpetra_Exceptions.hpp"
#include "Xpetra_MapFactory.hpp"

namespace Xpetra {





#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Node>
StridedMap<Node>::
#endif
StridedMap(UnderlyingLib                                 xlib,
           global_size_t                                 numGlobalElements,
           GlobalOrdinal                                 indexBase,
           std::vector<size_t>&                          stridingInfo,
           const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
           LocalOrdinal                                  stridedBlockId,      // FIXME (mfh 03 Sep 2014) This breaks for unsigned LocalOrdinal
           GlobalOrdinal                                 offset,
           LocalGlobal                                   lg)
    : stridingInfo_(stridingInfo), stridedBlockId_(stridedBlockId), offset_(offset), indexBase_(indexBase)
{
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    using MapFactory_t = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>;
#else
    using MapFactory_t = Xpetra::MapFactory<Node>;
#endif
    
    size_t blkSize = getFixedBlockSize();

    TEUCHOS_TEST_FOR_EXCEPTION(stridingInfo.size() == 0,
                               Exceptions::RuntimeError,
                               "StridedMap::StridedMap: stridingInfo not valid: stridingInfo.size() = 0?");

    TEUCHOS_TEST_FOR_EXCEPTION(numGlobalElements == Teuchos::OrdinalTraits<global_size_t>::invalid(),
                               std::invalid_argument,
                               "StridedMap::StridedMap: numGlobalElements is invalid");

    TEUCHOS_TEST_FOR_EXCEPTION(numGlobalElements % blkSize != 0,
                               Exceptions::RuntimeError,
                               "StridedMap::StridedMap: stridingInfo not valid: getFixedBlockSize "
                               "is not an integer multiple of numGlobalElements.");

    if(stridedBlockId != -1)
    {
        TEUCHOS_TEST_FOR_EXCEPTION(stridingInfo.size() < static_cast<size_t>(stridedBlockId),
                                   Exceptions::RuntimeError,
                                   "StridedTpetraMap::StridedTpetraMap: "
                                   "stridedBlockId > stridingInfo.size()");
    }

    // Try to create a shortcut
    if(blkSize != 1 || offset_ != 0)
    {
        // check input data and reorganize map
        global_size_t numGlobalNodes = numGlobalElements / blkSize;

        // build an equally distributed node map
        RCP<Map>      nodeMap       = MapFactory_t::Build(xlib, numGlobalNodes, indexBase, comm, lg);
        global_size_t numLocalNodes = nodeMap->getNodeNumElements();

        // translate local node ids to local dofs
        size_t nStridedOffset = 0;
        size_t nDofsPerNode   = blkSize;      // dofs per node for local striding block
        if(stridedBlockId > -1)
        {
            for(int j = 0; j < stridedBlockId; j++) 
            { 
                nStridedOffset += stridingInfo_[ j ]; 
            }

            nDofsPerNode      = stridingInfo_[ stridedBlockId ];
            numGlobalElements = numGlobalNodes * Teuchos::as<global_size_t>(nDofsPerNode);
        }
        size_t numLocalElements = numLocalNodes * Teuchos::as<size_t>(nDofsPerNode);

        std::vector<GlobalOrdinal> dofgids(numLocalElements);
        for(LocalOrdinal i = 0; i < Teuchos::as<LocalOrdinal>(numLocalNodes); i++)
        {
            GlobalOrdinal nodeGID = nodeMap->getGlobalElement(i);

            for(size_t j = 0; j < nDofsPerNode; j++)
            {
                dofgids[ i * nDofsPerNode + j ] = indexBase_ + offset_
                                                  + (nodeGID - indexBase_) * Teuchos::as<GlobalOrdinal>(blkSize)
                                                  + Teuchos::as<GlobalOrdinal>(nStridedOffset + j);
            }
        }

        map_ = MapFactory_t::Build(xlib, numGlobalElements, dofgids, indexBase, comm);

        if(stridedBlockId == -1)
        {
            TEUCHOS_TEST_FOR_EXCEPTION(getNodeNumElements() != Teuchos::as<size_t>(nodeMap->getNodeNumElements() * nDofsPerNode),
                                       Exceptions::RuntimeError,
                                       "StridedTpetraMap::StridedTpetraMap: wrong distribution of dofs among processors.");

            TEUCHOS_TEST_FOR_EXCEPTION(getGlobalNumElements()
                                         != Teuchos::as<size_t>(nodeMap->getGlobalNumElements() * nDofsPerNode),
                                       Exceptions::RuntimeError,
                                       "StridedTpetraMap::StridedTpetraMap: wrong distribution of dofs among processors.");
        }
        else
        {
            size_t nDofsInStridedBlock = stridingInfo[ stridedBlockId ];
            TEUCHOS_TEST_FOR_EXCEPTION(getNodeNumElements()
                                         != Teuchos::as<size_t>(nodeMap->getNodeNumElements() * nDofsInStridedBlock),
                                       Exceptions::RuntimeError,
                                       "StridedTpetraMap::StridedTpetraMap: wrong distribution of dofs among processors.");

            TEUCHOS_TEST_FOR_EXCEPTION(getGlobalNumElements()
                                         != Teuchos::as<size_t>(nodeMap->getGlobalNumElements() * nDofsInStridedBlock),
                                       Exceptions::RuntimeError,
                                       "StridedTpetraMap::StridedTpetraMap: wrong distribution of dofs among processors.");
        }
    }
    else
    {
        map_ = MapFactory_t::Build(xlib, numGlobalElements, indexBase, comm, lg);
    }

    TEUCHOS_TEST_FOR_EXCEPTION(CheckConsistency() == false, Exceptions::RuntimeError, "StridedTpetraMap::StridedTpetraMap: CheckConsistency() == false");
}




#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Node>
StridedMap<Node>::
#endif
StridedMap(UnderlyingLib                                 xlib,
           global_size_t                                 numGlobalElements,
           size_t                                        numLocalElements,
           GlobalOrdinal                                 indexBase,
           std::vector<size_t>&                          stridingInfo,
           const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
           LocalOrdinal                                  stridedBlockId,
           GlobalOrdinal                                 offset)
    : stridingInfo_(stridingInfo), stridedBlockId_(stridedBlockId), offset_(offset), indexBase_(indexBase)
{
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    using MapFactory_t = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>;
#else
    using MapFactory_t = Xpetra::MapFactory<Node>;
#endif
    
    size_t blkSize = getFixedBlockSize();
    TEUCHOS_TEST_FOR_EXCEPTION(stridingInfo.size() == 0,
                               Exceptions::RuntimeError,
                               "StridedMap::StridedMap: stridingInfo not valid: stridingInfo.size() = 0?");
    if(numGlobalElements != Teuchos::OrdinalTraits<global_size_t>::invalid())
    {
        TEUCHOS_TEST_FOR_EXCEPTION(numGlobalElements % blkSize != 0,
                                   Exceptions::RuntimeError,
                                   "StridedMap::StridedMap: stridingInfo not valid: getFixedBlockSize is not an integer "
                                   "multiple of numGlobalElements.");
#ifdef HAVE_XPETRA_DEBUG
        // We have to do this check ourselves, as we don't necessarily construct the full Tpetra map
        global_size_t sumLocalElements;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, Teuchos::as<global_size_t>(numLocalElements), Teuchos::outArg(sumLocalElements));

        TEUCHOS_TEST_FOR_EXCEPTION(sumLocalElements != numGlobalElements,
                                   std::invalid_argument,
                                   "StridedMap::StridedMap: sum of numbers of local elements is different from the provided "
                                   "number of global elements.");
#endif
    }

    TEUCHOS_TEST_FOR_EXCEPTION(
      numLocalElements % blkSize != 0,
      Exceptions::RuntimeError,
      "StridedMap::StridedMap: stridingInfo not valid: getFixedBlockSize is not an integer multiple of numLocalElements.");

    if(stridedBlockId != -1)
    {
        TEUCHOS_TEST_FOR_EXCEPTION(stridingInfo.size() < Teuchos::as<size_t>(stridedBlockId),
                                   Exceptions::RuntimeError,
                                   "StridedTpetraMap::StridedTpetraMap: stridedBlockId > stridingInfo.size()");
    }

    // Try to create a shortcut
    if(blkSize != 1 || offset_ != 0)
    {
        // check input data and reorganize map
        global_size_t numGlobalNodes = Teuchos::OrdinalTraits<global_size_t>::invalid();
        if(numGlobalElements != Teuchos::OrdinalTraits<global_size_t>::invalid())
        {
            numGlobalNodes = numGlobalElements / blkSize;
        }
        global_size_t numLocalNodes = numLocalElements / blkSize;

        // build an equally distributed node map
        RCP<Map> nodeMap = MapFactory_t::Build(xlib, numGlobalNodes, numLocalNodes, indexBase, comm);

        // translate local node ids to local dofs
        size_t nStridedOffset = 0;
        size_t nDofsPerNode   = blkSize;      // dofs per node for local striding block
        if(stridedBlockId > -1)
        {
            for(int j = 0; j < stridedBlockId; j++) 
            { 
                nStridedOffset += stridingInfo_[ j ]; 
            }

            nDofsPerNode      = stridingInfo_[ stridedBlockId ];
            numGlobalElements = nodeMap->getGlobalNumElements() * Teuchos::as<global_size_t>(nDofsPerNode);
        }
        numLocalElements = numLocalNodes * Teuchos::as<size_t>(nDofsPerNode);

        std::vector<GlobalOrdinal> dofgids(numLocalElements);
        for(LocalOrdinal i = 0; i < Teuchos::as<LocalOrdinal>(numLocalNodes); i++)
        {
            GlobalOrdinal nodeGID = nodeMap->getGlobalElement(i);

            for(size_t j = 0; j < nDofsPerNode; j++)
            {
                dofgids[ i * nDofsPerNode + j ] = indexBase_ + offset_
                                                  + (nodeGID - indexBase_) * Teuchos::as<GlobalOrdinal>(blkSize)
                                                  + Teuchos::as<GlobalOrdinal>(nStridedOffset + j);
            }
        }

        map_ = MapFactory_t::Build(xlib, numGlobalElements, dofgids, indexBase, comm);

        if(stridedBlockId == -1)
        {
            TEUCHOS_TEST_FOR_EXCEPTION(getNodeNumElements() != Teuchos::as<size_t>(nodeMap->getNodeNumElements() * nDofsPerNode),
                                       Exceptions::RuntimeError,
                                       "StridedTpetraMap::StridedTpetraMap: wrong distribution of dofs among processors.");

            TEUCHOS_TEST_FOR_EXCEPTION(getGlobalNumElements()
                                       != Teuchos::as<size_t>(nodeMap->getGlobalNumElements() * nDofsPerNode),
                                       Exceptions::RuntimeError,
                                       "StridedTpetraMap::StridedTpetraMap: wrong distribution of dofs among processors.");
        }
        else
        {
            int nDofsInStridedBlock = stridingInfo[ stridedBlockId ];

            TEUCHOS_TEST_FOR_EXCEPTION(getNodeNumElements()
                                       != Teuchos::as<size_t>(nodeMap->getNodeNumElements() * nDofsInStridedBlock),
                                       Exceptions::RuntimeError,
                                       "StridedTpetraMap::StridedTpetraMap: wrong distribution of dofs among processors.");

            TEUCHOS_TEST_FOR_EXCEPTION(getGlobalNumElements()
                                       != Teuchos::as<size_t>(nodeMap->getGlobalNumElements() * nDofsInStridedBlock),
                                       Exceptions::RuntimeError,
                                       "StridedTpetraMap::StridedTpetraMap: wrong distribution of dofs among processors.");
        }
    }
    else
    {
        map_ = MapFactory_t::Build(xlib, numGlobalElements, numLocalElements, indexBase, comm);
    }

    TEUCHOS_TEST_FOR_EXCEPTION(CheckConsistency() == false, Exceptions::RuntimeError, "StridedTpetraMap::StridedTpetraMap: CheckConsistency() == false");
}




#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Node>
StridedMap<Node>::
#endif
StridedMap(UnderlyingLib                                  xlib,
           global_size_t                                  numGlobalElements,
           const Teuchos::ArrayView<const GlobalOrdinal>& elementList,
           GlobalOrdinal                                  indexBase,
           std::vector<size_t>&                           stridingInfo,
           const Teuchos::RCP<const Teuchos::Comm<int>>&  comm,
           LocalOrdinal                                   stridedBlockId)
    : stridingInfo_(stridingInfo), stridedBlockId_(stridedBlockId), indexBase_(indexBase)
{
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    using MapFactory_t = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>;
#else
    using MapFactory_t = Xpetra::MapFactory<Node>;
#endif
    
    size_t blkSize = getFixedBlockSize();

    TEUCHOS_TEST_FOR_EXCEPTION(stridingInfo.size() == 0,
                               Exceptions::RuntimeError,
                               "StridedMap::StridedMap: stridingInfo not valid: stridingInfo.size() = 0?");
    if(stridedBlockId != -1)
        TEUCHOS_TEST_FOR_EXCEPTION(stridingInfo.size() < Teuchos::as<size_t>(stridedBlockId),
                                   Exceptions::RuntimeError,
                                   "StridedTpetraMap::StridedTpetraMap: stridedBlockId > stridingInfo.size()");
    if(numGlobalElements != Teuchos::OrdinalTraits<global_size_t>::invalid())
    {
        TEUCHOS_TEST_FOR_EXCEPTION(numGlobalElements % blkSize != 0,
                                   Exceptions::RuntimeError,
                                   "StridedMap::StridedMap: stridingInfo not valid: getFixedBlockSize is not an integer "
                                   "multiple of numGlobalElements.");
#ifdef HAVE_XPETRA_DEBUG
        // We have to do this check ourselves, as we don't necessarily construct the full Tpetra map
        global_size_t sumLocalElements, numLocalElements = elementList.size();
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, numLocalElements, Teuchos::outArg(sumLocalElements));
        TEUCHOS_TEST_FOR_EXCEPTION(sumLocalElements != numGlobalElements,
                                   std::invalid_argument,
                                   "StridedMap::StridedMap: sum of numbers of local elements is different from the provided "
                                   "number of global elements.");
#endif
    }

    if(stridedBlockId == -1)
    {
        // numGlobalElements can be -1! FIXME
        // TEUCHOS_TEST_FOR_EXCEPTION(numGlobalElements  % blkSize != 0, Exceptions::RuntimeError,
        // "StridedMap::StridedMap: stridingInfo not valid: getFixedBlockSize is not an integer multiple of
        // numGlobalElements.");
        TEUCHOS_TEST_FOR_EXCEPTION(elementList.size() % blkSize != 0,
                                   Exceptions::RuntimeError,
                                   "StridedMap::StridedMap: stridingInfo not valid: getFixedBlockSize is not an integer "
                                   "multiple of elementList.size().");
    }
    else
    {
        // numGlobalElements can be -1! FIXME
        // TEUCHOS_TEST_FOR_EXCEPTION(numGlobalElements  % stridingInfo[stridedBlockId] != 0, Exceptions::RuntimeError,
        // "StridedMap::StridedMap: stridingInfo not valid: stridingBlockInfo[stridedBlockId] is not an integer multiple of
        // numGlobalElements.");
        TEUCHOS_TEST_FOR_EXCEPTION(elementList.size() % stridingInfo[ stridedBlockId ] != 0,
                                   Exceptions::RuntimeError,
                                   "StridedMap::StridedMap: stridingInfo not valid: stridingBlockInfo[stridedBlockId] is not "
                                   "an integer multiple of elementList.size().");
    }

    map_ = MapFactory_t::Build(xlib, numGlobalElements, elementList, indexBase, comm);

    // calculate offset_

    // find minimum GID over all procs
    GlobalOrdinal minGidOnCurProc = Teuchos::OrdinalTraits<GlobalOrdinal>::max();
    for(Teuchos_Ordinal k = 0; k < elementList.size(); k++)      // TODO fix occurence of Teuchos_Ordinal
    {
        if(elementList[ k ] < minGidOnCurProc)
        {
            minGidOnCurProc = elementList[ k ];
        }
    }

    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, minGidOnCurProc, Teuchos::outArg(offset_));

    // calculate striding index
    size_t nStridedOffset = 0;
    for(int j = 0; j < stridedBlockId; j++) 
    {
      nStridedOffset += stridingInfo[ j ];
    }
    const GlobalOrdinal goStridedOffset = Teuchos::as<GlobalOrdinal>(nStridedOffset);

    // adapt offset_
    offset_ -= goStridedOffset + indexBase_;

    TEUCHOS_TEST_FOR_EXCEPTION(CheckConsistency() == false, Exceptions::RuntimeError, "StridedTpetraMap::StridedTpetraMap: CheckConsistency() == false");
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Node>
StridedMap<Node>::
#endif
StridedMap(const RCP<const Map>& map,
           std::vector<size_t>&  stridingInfo,
           GlobalOrdinal         /* indexBase */,
           LocalOrdinal          stridedBlockId,
           GlobalOrdinal         offset)
    : stridingInfo_(stridingInfo), 
      stridedBlockId_(stridedBlockId), 
      offset_(offset), 
      indexBase_(map->getIndexBase())
{
    // TAW: 11/24/15
    //      A strided map never can be built from a strided map. getMap always returns the underlying
    //      Xpetra::Map object which contains the data (either in a Xpetra::EpetraMapT or Xpetra::TpetraMap
    //      object)
    if(Teuchos::rcp_dynamic_cast<const StridedMap>(map) == Teuchos::null)
    {
        map_ = map;      // if map is not a strided map, just store it (standard case)
    }
    else
    {
        map_ = map->getMap();      // if map is also a strided map, store the underlying plain Epetra/Tpetra Xpetra map object
    }
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Node>
StridedMap<Node>::
#endif
~StridedMap() 
{
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
std::vector<size_t>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getStridingData() const
{
    return stridingInfo_;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
setStridingData(std::vector<size_t> stridingInfo)
{
    stridingInfo_ = stridingInfo;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
size_t
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getFixedBlockSize() const
{
    size_t blkSize = 0;
    for(std::vector<size_t>::const_iterator it = stridingInfo_.begin(); it != stridingInfo_.end(); ++it) 
    { 
        blkSize += *it;
    }
    return blkSize;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
LocalOrdinal
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getStridedBlockId() const
{
    return stridedBlockId_;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
bool
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
isStrided() const
{
    return stridingInfo_.size() > 1 ? true : false;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
bool
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
isBlocked() const
{
    return getFixedBlockSize() > 1 ? true : false;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
GlobalOrdinal
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getOffset() const
{
    return offset_;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
setOffset(GlobalOrdinal offset)
{
    offset_ = offset;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
size_t
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
GID2StridingBlockId(GlobalOrdinal gid) const
{
    GlobalOrdinal tgid = gid - offset_ - indexBase_;
    tgid               = tgid % getFixedBlockSize();

    size_t nStridedOffset = 0;
    size_t stridedBlockId = 0;
    for(size_t j = 0; j < stridingInfo_.size(); j++)
    {
        nStridedOffset += stridingInfo_[ j ];
        if(Teuchos::as<size_t>(tgid) < nStridedOffset)
        {
            stridedBlockId = j;
            break;
        }
    }
    return stridedBlockId;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Node>
RCP<const Xpetra::Map<Node>>
StridedMap<Node>::
#endif
getMap() const
{
    return map_;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
bool
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
CheckConsistency()
{
#ifndef HAVE_XPETRA_DEBUG
    return true;
#else
    if(getStridedBlockId() == -1)
    {
        // Strided map contains the full map
        if(getNodeNumElements() % getFixedBlockSize() != 0 ||      // number of local  elements is not a multiple of block size
           getGlobalNumElements() % getFixedBlockSize() != 0)      // number of global    -//-
            return false;
    }
    else
    {
        // Strided map contains only the partial map
        Teuchos::ArrayView<const GlobalOrdinal> dofGids = getNodeElementList();
        // std::sort(dofGids.begin(), dofGids.end());

        if(dofGids.size() == 0)      // special treatment for empty processors
        {
            return true;
        }

        if(dofGids.size() % stridingInfo_[ stridedBlockId_ ] != 0)
        {
            return false;
        }


        // Calculate nStridedOffset
        size_t nStridedOffset = 0;
        for(int j = 0; j < stridedBlockId_; j++) 
        {
            nStridedOffset += stridingInfo_[ j ]; 
        }

        const GlobalOrdinal goStridedOffset = Teuchos::as<GlobalOrdinal>(nStridedOffset);
        const GlobalOrdinal goZeroOffset    = (dofGids[ 0 ] - nStridedOffset - offset_ - indexBase_) / Teuchos::as<GlobalOrdinal>(getFixedBlockSize());

        GlobalOrdinal cnt = 0;
        for(size_t i = 0; 
            i < Teuchos::as<size_t>(dofGids.size()) / stridingInfo_[ stridedBlockId_ ]; 
            i += stridingInfo_[ stridedBlockId_ ])
        {
            const GlobalOrdinal first_gid = dofGids[ i ];

            // We expect this to be the same for all DOFs of the same node
            cnt = (first_gid - goStridedOffset - offset_ - indexBase_) / Teuchos::as<GlobalOrdinal>(getFixedBlockSize()) - goZeroOffset;

            // Loop over all DOFs that belong to current node
            for(size_t j = 0; j < stridingInfo_[ stridedBlockId_ ]; j++)
            {
                const GlobalOrdinal gid = dofGids[ i + j ];
                const GlobalOrdinal r   = (gid - Teuchos::as<GlobalOrdinal>(j) - goStridedOffset - offset_ - indexBase_)
                                          / Teuchos::as<GlobalOrdinal>(getFixedBlockSize())
                                          - goZeroOffset - cnt;
                // TAW 1/18/2016: We cannot use Teuchos::OrdinalTraits<GlobalOrdinal>::zero() ) here,
                //                If, e.g., GO=long long is disabled, OrdinalTraits<long long> is not available.
                //                But we instantiate stubs on GO=long long which might contain StridedMaps.
                //                These lead to compilation errors, then.
                if(0 != r)
                {
                    std::cout << "goZeroOffset   : " << goZeroOffset << std::endl
                              << "dofGids[0]     : " << dofGids[ 0 ] << std::endl
                              << "stridedOffset  : " << nStridedOffset << std::endl
                              << "offset_        : " << offset_ << std::endl
                              << "goStridedOffset: " << goStridedOffset << std::endl
                              << "getFixedBlkSize: " << getFixedBlockSize() << std::endl
                              << "gid: " << gid << " GID: " << r << std::endl;

                    return false;
                }
            }
        }
    }

    return true;
#endif
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
global_size_t
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getGlobalNumElements() const
{
    return map_->getGlobalNumElements();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
size_t
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getNodeNumElements() const
{
    return map_->getNodeNumElements();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
GlobalOrdinal
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getIndexBase() const
{
    return map_->getIndexBase();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
LocalOrdinal
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getMinLocalIndex() const
{
    return map_->getMinLocalIndex();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
LocalOrdinal
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getMaxLocalIndex() const
{
    return map_->getMaxLocalIndex();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
GlobalOrdinal
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getMinGlobalIndex() const
{
    return map_->getMinGlobalIndex();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
GlobalOrdinal
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getMaxGlobalIndex() const
{
    return map_->getMaxGlobalIndex();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
GlobalOrdinal
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getMinAllGlobalIndex() const
{
    return map_->getMinAllGlobalIndex();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
GlobalOrdinal
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getMaxAllGlobalIndex() const
{
    return map_->getMaxAllGlobalIndex();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
LocalOrdinal
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getLocalElement(GlobalOrdinal globalIndex) const
{
    return map_->getLocalElement(globalIndex);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
GlobalOrdinal
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getGlobalElement(LocalOrdinal localIndex) const
{
    return map_->getGlobalElement(localIndex);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
LookupStatus
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getRemoteIndexList(const Teuchos::ArrayView<const GlobalOrdinal>& GIDList,
                   const Teuchos::ArrayView<int>&                 nodeIDList,
                   const Teuchos::ArrayView<LocalOrdinal>&        LIDList) const
{
    return map_->getRemoteIndexList(GIDList, nodeIDList, LIDList);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
LookupStatus
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getRemoteIndexList(const Teuchos::ArrayView<const GlobalOrdinal>& GIDList, 
                   const Teuchos::ArrayView<int>&                 nodeIDList) const
{
    return map_->getRemoteIndexList(GIDList, nodeIDList);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
Teuchos::ArrayView<const GlobalOrdinal>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getNodeElementList() const
{
    return map_->getNodeElementList();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
bool
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
isNodeLocalElement(LocalOrdinal localIndex) const
{
    return map_->isNodeLocalElement(localIndex);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
bool
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
isNodeGlobalElement(GlobalOrdinal globalIndex) const
{
    return map_->isNodeGlobalElement(globalIndex);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
bool
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
isContiguous() const
{
    return map_->isContiguous();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
bool
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
isDistributed() const
{
    return map_->isDistributed();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
bool
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
isCompatible(const Map& map) const
{
    return map_->isCompatible(map);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
bool
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
isSameAs(const Map& map) const
{
    return map_->isSameAs(map);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
Teuchos::RCP<const Teuchos::Comm<int>>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
getComm() const
{
    return map_->getComm();
}




#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>>
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Node>
RCP<const Xpetra::Map<Node>>
StridedMap<Node>::
#endif
removeEmptyProcesses() const
{
    return map_->removeEmptyProcesses();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>>
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Node>
RCP<const Xpetra::Map<Node>>
StridedMap<Node>::
#endif
replaceCommWithSubset(const Teuchos::RCP<const Teuchos::Comm<int>>& newComm) const
{
    return map_->replaceCommWithSubset(newComm);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
std::string
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
description() const
{
    return map_->description();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
describe(Teuchos::FancyOStream& out, const Teuchos::EVerbosityLevel verbLevel) const
{
    map_->describe(out, verbLevel);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
UnderlyingLib
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
StridedMap<LocalOrdinal, GlobalOrdinal, Node>::
#else
StridedMap<Node>::
#endif
lib() const
{
    return map_->lib();
}



}      // namespace Xpetra



#endif      // XPETRA_STRIDEDMAP_DEF_HPP


