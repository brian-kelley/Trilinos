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
#ifndef XPETRA_MAPEXTRACTOR_DEF_HPP_
#define XPETRA_MAPEXTRACTOR_DEF_HPP_

#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_VectorFactory.hpp>
#include <Xpetra_BlockedMultiVector.hpp>

#include <Xpetra_MapExtractor_decl.hpp>

namespace Xpetra {


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    template <class Scalar, class Node>
    MapExtractor<Scalar, Node>::
#endif
    MapExtractor(const RCP<const Map>& fullmap, const std::vector<RCP<const Map> >& maps, bool bThyraMode)
    {
      map_ = Teuchos::rcp(new BlockedMap(fullmap, maps, bThyraMode));
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    template <class Scalar, class Node>
    MapExtractor<Scalar, Node>::
#endif
    MapExtractor(const std::vector<RCP<const Map> >& maps, const std::vector<RCP<const Map> >& thyramaps)
    {
      map_ = Teuchos::rcp(new BlockedMap(maps, thyramaps));
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    template <class Scalar, class Node>
    MapExtractor<Scalar, Node>::
#endif
    MapExtractor(const Teuchos::RCP<const BlockedMap>& map)
        : map_(map)
    {}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    template <class Scalar, class Node>
    MapExtractor<Scalar, Node>::
#endif
    MapExtractor(const MapExtractor& input)
    {
      map_ = Teuchos::rcp(new BlockedMap(*(input.getBlockedMap())));
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    template <class Scalar, class Node>
    MapExtractor<Scalar, Node>::
#endif
    ~MapExtractor()
    {
      map_ = Teuchos::null;
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    MapExtractor<Scalar, Node>::
#endif
    ExtractVector(const Vector& full, size_t block, Vector& partial) const
    {
      XPETRA_TEST_FOR_EXCEPTION(block >= map_->getNumMaps(), std::out_of_range,
            "ExtractVector: Error, block = " << block << " is too big. The MapExtractor only contains " << map_->getNumMaps() << " partial blocks.");
      XPETRA_TEST_FOR_EXCEPTION(map_->getMap(block,false) == null, Xpetra::Exceptions::RuntimeError,
            "ExtractVector: map_->getMap(" << block << ",false) is null");

      partial.doImport(full, *(map_->getImporter(block)), Xpetra::INSERT);
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    MapExtractor<Scalar, Node>::
#endif
    ExtractVector(const MultiVector& full, size_t block, MultiVector& partial) const
    {
        XPETRA_TEST_FOR_EXCEPTION(block >= map_->getNumMaps(),
                                  std::out_of_range,
                                  "ExtractVector: Error, block = " << block << " is too big. The MapExtractor only contains " << map_->getNumMaps()
                                                                   << " partial blocks.");
        XPETRA_TEST_FOR_EXCEPTION(
          map_->getMap(block, false) == null, Xpetra::Exceptions::RuntimeError, "ExtractVector: map_->getMap(" << block << ",false) is null");

        partial.doImport(full, *(map_->getImporter(block)), Xpetra::INSERT);
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    MapExtractor<Scalar, Node>::
#endif
    ExtractVector(RCP<const Vector>& full, size_t block, RCP<Vector>& partial) const
    {
        ExtractVector(*full, block, *partial);
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    MapExtractor<Scalar, Node>::
#endif
    ExtractVector(RCP<Vector>& full, size_t block, RCP<Vector>& partial) const
    {
        ExtractVector(*full, block, *partial);
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    MapExtractor<Scalar, Node>::
#endif
    ExtractVector(RCP<const MultiVector>& full, size_t block, RCP<MultiVector>& partial) const
    {
        ExtractVector(*full, block, *partial);
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    MapExtractor<Scalar, Node>::
#endif
    ExtractVector(RCP<MultiVector>& full, size_t block, RCP<MultiVector>& partial) const
    {
        ExtractVector(*full, block, *partial);
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    RCP<Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    ExtractVector(RCP<const Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& full, size_t block, bool bThyraMode) const
#else
    template <class Scalar, class Node>
    RCP<Xpetra::Vector<Scalar,Node> >
    MapExtractor<Scalar, Node>::
    ExtractVector(RCP<const Xpetra::Vector<Scalar,Node> >& full, size_t block, bool bThyraMode) const
#endif
    {
        XPETRA_TEST_FOR_EXCEPTION(block >= map_->getNumMaps(),
                                  std::out_of_range,
                                  "ExtractVector: Error, block = " << block << " is too big. The MapExtractor only contains " << map_->getNumMaps()
                                                                   << " partial blocks.");
        XPETRA_TEST_FOR_EXCEPTION(
          map_->getMap(block, false) == null, Xpetra::Exceptions::RuntimeError, "ExtractVector: map_->getMap(" << block << ",false) is null");
        // first extract partial vector from full vector (using xpetra style GIDs)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        const RCP<Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > vv = Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(getMap(block, false), false);
#else
        const RCP<Xpetra::Vector<Scalar,Node> > vv = Xpetra::VectorFactory<Scalar,Node>::Build(getMap(block, false), false);
#endif
        ExtractVector(*full, block, *vv);
        if(bThyraMode == false)
            return vv;
        TEUCHOS_TEST_FOR_EXCEPTION(map_->getThyraMode() == false && bThyraMode == true,
                                   Xpetra::Exceptions::RuntimeError,
                                   "MapExtractor::ExtractVector: ExtractVector in Thyra-style numbering only possible if MapExtractor has been "
                                   "created using Thyra-style numbered submaps.");
        vv->replaceMap(getMap(block, true));      // switch to Thyra-style map
        return vv;
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    RCP<Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    ExtractVector(RCP<Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& full, size_t block, bool bThyraMode) const
#else
    template <class Scalar, class Node>
    RCP<Xpetra::Vector<Scalar, Node> >
    MapExtractor<Scalar, Node>::
    ExtractVector(RCP<Xpetra::Vector<Scalar, Node> >& full, size_t block, bool bThyraMode) const
#endif
    {
        XPETRA_TEST_FOR_EXCEPTION(block >= map_->getNumMaps(),
                                  std::out_of_range,
                                  "ExtractVector: Error, block = " << block << " is too big. The MapExtractor only contains " << map_->getNumMaps()
                                                                   << " partial blocks.");
        XPETRA_TEST_FOR_EXCEPTION(
          map_->getMap(block, false) == null, Xpetra::Exceptions::RuntimeError, "ExtractVector: map_->getmap(" << block << ",false) is null");
        // first extract partial vector from full vector (using xpetra style GIDs)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
const RCP<Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > vv = Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(getMap(block, false), false);
#else
const RCP<Xpetra::Vector<Scalar,Node> > vv = Xpetra::VectorFactory<Scalar,Node>::Build(getMap(block, false), false);
#endif

        ExtractVector(*full, block, *vv);
        if(bThyraMode == false)
            return vv;
        TEUCHOS_TEST_FOR_EXCEPTION(map_->getThyraMode() == false && bThyraMode == true,
                                   Xpetra::Exceptions::RuntimeError,
                                   "MapExtractor::ExtractVector: ExtractVector in Thyra-style numbering only possible if MapExtractor has been "
                                   "created using Thyra-style numbered submaps.");
        vv->replaceMap(getMap(block, true));      // switch to Thyra-style map
        return vv;
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    ExtractVector(RCP<const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& full, size_t block, bool bThyraMode) const
#else
    template <class Scalar, class Node>
    RCP<Xpetra::MultiVector<Scalar, Node> >
    MapExtractor<Scalar, Node>::
    ExtractVector(RCP<const Xpetra::MultiVector<Scalar,Node> >& full, size_t block, bool bThyraMode) const
#endif
    {
        XPETRA_TEST_FOR_EXCEPTION(block >= map_->getNumMaps(),
                                  std::out_of_range,
                                  "ExtractVector: Error, block = " << block << " is too big. The MapExtractor only contains " << map_->getNumMaps()
                                                                   << " partial blocks.");
        XPETRA_TEST_FOR_EXCEPTION(
          map_->getMap(block, false) == null, Xpetra::Exceptions::RuntimeError, "ExtractVector: map_->getmap(" << block << ",false) is null");
        RCP<const BlockedMultiVector> bfull = Teuchos::rcp_dynamic_cast<const BlockedMultiVector>(full);
        if(bfull.is_null() == true)
        {
            // standard case: full is not of type BlockedMultiVector
            // first extract partial vector from full vector (using xpetra style GIDs)
            const RCP<MultiVector> vv = MultiVectorFactory::Build(getMap(block, false), full->getNumVectors(), false);
            // if(bThyraMode == false) {
            //  ExtractVector(*full, block, *vv);
            //  return vv;
            //} else {
            RCP<const Map>   oldThyMapFull   = full->getMap();      // temporarely store map of full
            RCP<MultiVector> rcpNonConstFull = Teuchos::rcp_const_cast<MultiVector>(full);
            rcpNonConstFull->replaceMap(map_->getImporter(block)->getSourceMap());
            ExtractVector(*rcpNonConstFull, block, *vv);
            TEUCHOS_TEST_FOR_EXCEPTION(map_->getThyraMode() == false && bThyraMode == true,
                                       Xpetra::Exceptions::RuntimeError,
                                       "MapExtractor::ExtractVector: ExtractVector in Thyra-style numbering only possible if MapExtractor has been "
                                       "created using Thyra-style numbered submaps.");
            if(bThyraMode == true)
                vv->replaceMap(getMap(block, true));      // switch to Thyra-style map
            rcpNonConstFull->replaceMap(oldThyMapFull);
            return vv;
            //}
        }
        else
        {
            // special case: full is of type BlockedMultiVector
            XPETRA_TEST_FOR_EXCEPTION(map_->getNumMaps() != bfull->getBlockedMap()->getNumMaps(),
                                      Xpetra::Exceptions::RuntimeError,
                                      "ExtractVector: Number of blocks in map extractor is " << map_->getNumMaps() << " but should be "
                                                                                             << bfull->getBlockedMap()->getNumMaps()
                                                                                             << " (number of blocks in BlockedMultiVector)");
            return bfull->getMultiVector(block, bThyraMode);
        }
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    ExtractVector(RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& full, size_t block, bool bThyraMode) const
#else
    template <class Scalar, class Node>
    RCP<Xpetra::MultiVector<Scalar,Node> >
    MapExtractor<Scalar, Node>::
    ExtractVector(RCP<Xpetra::MultiVector<Scalar,Node> >& full, size_t block, bool bThyraMode) const
#endif
    {
        XPETRA_TEST_FOR_EXCEPTION(block >= map_->getNumMaps(),
                                  std::out_of_range,
                                  "ExtractVector: Error, block = " << block << " is too big. The MapExtractor only contains " << map_->getNumMaps()
                                                                   << " partial blocks.");
        XPETRA_TEST_FOR_EXCEPTION(
          map_->getMap(block, false) == null, Xpetra::Exceptions::RuntimeError, "ExtractVector: map_->getmap(" << block << ",false) is null");
        RCP<BlockedMultiVector> bfull = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(full);
        if(bfull.is_null() == true)
        {
            // standard case: full is not of type BlockedMultiVector
            // first extract partial vector from full vector (using xpetra style GIDs)
            const RCP<MultiVector> vv = MultiVectorFactory::Build(getMap(block, false), full->getNumVectors(), false);
            // if(bThyraMode == false) {
            //  ExtractVector(*full, block, *vv);
            //  return vv;
            //} else {
            RCP<const Map> oldThyMapFull = full->getMap();      // temporarely store map of full
            full->replaceMap(map_->getImporter(block)->getSourceMap());
            ExtractVector(*full, block, *vv);
            TEUCHOS_TEST_FOR_EXCEPTION(map_->getThyraMode() == false && bThyraMode == true,
                                       Xpetra::Exceptions::RuntimeError,
                                       "MapExtractor::ExtractVector: ExtractVector in Thyra-style numbering only possible if MapExtractor has been "
                                       "created using Thyra-style numbered submaps.");
            if(bThyraMode == true)
                vv->replaceMap(getMap(block, true));      // switch to Thyra-style map
            full->replaceMap(oldThyMapFull);
            return vv;
            //}
        }
        else
        {
            // special case: full is of type BlockedMultiVector
            XPETRA_TEST_FOR_EXCEPTION(map_->getNumMaps() != bfull->getBlockedMap()->getNumMaps(),
                                      Xpetra::Exceptions::RuntimeError,
                                      "ExtractVector: Number of blocks in map extractor is " << map_->getNumMaps() << " but should be "
                                                                                             << bfull->getBlockedMap()->getNumMaps()
                                                                                             << " (number of blocks in BlockedMultiVector)");
            return bfull->getMultiVector(block, bThyraMode);
        }
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    ExtractVector(RCP<const Xpetra::BlockedMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& full, size_t block, bool bThyraMode) const
#else
    template <class Scalar, class Node>
    RCP<Xpetra::MultiVector<Scalar,Node> >
    MapExtractor<Scalar, Node>::
    ExtractVector(RCP<const Xpetra::BlockedMultiVector<Scalar,Node> >& full, size_t block, bool bThyraMode) const
#endif
    {
        XPETRA_TEST_FOR_EXCEPTION(block >= map_->getNumMaps(),
                                  std::out_of_range,
                                  "ExtractVector: Error, block = " << block << " is too big. The MapExtractor only contains " << map_->getNumMaps()
                                                                   << " partial blocks.");
        XPETRA_TEST_FOR_EXCEPTION(
          map_->getMap(block, false) == null, Xpetra::Exceptions::RuntimeError, "ExtractVector: map_->getmap(" << block << ",false) is null");
        XPETRA_TEST_FOR_EXCEPTION(map_->getNumMaps() != full->getBlockedMap()->getNumMaps(),
                                  Xpetra::Exceptions::RuntimeError,
                                  "ExtractVector: Number of blocks in map extractor is " << map_->getNumMaps() << " but should be "
                                                                                         << full->getBlockedMap()->getNumMaps()
                                                                                         << " (number of blocks in BlockedMultiVector)");
        Teuchos::RCP<MultiVector> vv = full->getMultiVector(block, bThyraMode);
        return vv;
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    ExtractVector(RCP<Xpetra::BlockedMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>& full, size_t block, bool bThyraMode) const
#else
    template <class Scalar, class Node>
    RCP<Xpetra::MultiVector<Scalar,Node> >
    MapExtractor<Scalar, Node>::
    ExtractVector(RCP<Xpetra::BlockedMultiVector<Scalar, Node>>& full, size_t block, bool bThyraMode) const
#endif
    {
        XPETRA_TEST_FOR_EXCEPTION(block >= map_->getNumMaps(),
                                  std::out_of_range,
                                  "ExtractVector: Error, block = " << block << " is too big. The MapExtractor only contains " << map_->getNumMaps()
                                                                   << " partial blocks.");
        XPETRA_TEST_FOR_EXCEPTION(
          map_->getMap(block, false) == null, Xpetra::Exceptions::RuntimeError, "ExtractVector: map_->getmap(" << block << ",false) is null");
        XPETRA_TEST_FOR_EXCEPTION(map_->getNumMaps() != full->getBlockedMap()->getNumMaps(),
                                  Xpetra::Exceptions::RuntimeError,
                                  "ExtractVector: Number of blocks in map extractor is " << map_->getNumMaps() << " but should be "
                                                                                         << full->getBlockedMap()->getNumMaps()
                                                                                         << " (number of blocks in BlockedMultiVector)");
        Teuchos::RCP<MultiVector> vv = full->getMultiVector(block, bThyraMode);
        return vv;
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    InsertVector(const Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& partial, size_t block, Vector& full, bool bThyraMode) const
#else
    MapExtractor<Scalar, Node>::
    InsertVector(const Xpetra::Vector<Scalar,Node>& partial, size_t block, Vector& full, bool bThyraMode) const
#endif
    {
        XPETRA_TEST_FOR_EXCEPTION(block >= map_->getNumMaps(),
                                  std::out_of_range,
                                  "ExtractVector: Error, block = " << block << " is too big. The MapExtractor only contains " << map_->getNumMaps()
                                                                   << " partial blocks.");
        XPETRA_TEST_FOR_EXCEPTION(
          map_->getMap(block, false) == null, Xpetra::Exceptions::RuntimeError, "ExtractVector: map_->getmap(" << block << ",false) is null");
        XPETRA_TEST_FOR_EXCEPTION(map_->getThyraMode() == false && bThyraMode == true,
                                  Xpetra::Exceptions::RuntimeError,
                                  "MapExtractor::InsertVector: InsertVector in Thyra-style numbering only possible if MapExtractor has been created "
                                  "using Thyra-style numbered submaps.");
        if(bThyraMode)
        {
            // NOTE: the importer objects in the BlockedMap are always using Xpetra GIDs (or Thyra style Xpetra GIDs)
            // The source map corresponds to the full map (in Xpetra GIDs) starting with GIDs from zero. The GIDs are consecutive in Thyra mode
            // The target map is the partial map (in the corresponding Xpetra GIDs)

            // TODO can we skip the Export call in special cases (i.e. Src = Target map, same length, etc...)

            // store original GIDs (could be Thyra GIDs)
            RCP<const MultiVector> rcpPartial         = Teuchos::rcpFromRef(partial);
            RCP<MultiVector>       rcpNonConstPartial = Teuchos::rcp_const_cast<MultiVector>(rcpPartial);
            RCP<const Map>         oldThyMapPartial   = rcpNonConstPartial->getMap();      // temporarely store map of partial
            RCP<const Map>         oldThyMapFull      = full.getMap();                     // temporarely store map of full

            // check whether getMap(block,false) is identical to target map of importer
            XPETRA_TEST_FOR_EXCEPTION(map_->getMap(block, false)->isSameAs(*(map_->getImporter(block)->getTargetMap())) == false,
                                      Xpetra::Exceptions::RuntimeError,
                                      "MapExtractor::InsertVector: InsertVector in Thyra-style mode: Xpetra GIDs of partial vector are not identical "
                                      "to target Map of Importer. This should not be.");

            // XPETRA_TEST_FOR_EXCEPTION(full.getMap()->isSameAs(*(map_->getImporter(block)->getSourceMap()))==false,
            // Xpetra::Exceptions::RuntimeError,
            //           "MapExtractor::InsertVector: InsertVector in Thyra-style mode: Xpetra GIDs of full vector are not identical to source Map of
            //           Importer. This should not be.");

            rcpNonConstPartial->replaceMap(getMap(block, false));           // temporarely switch to xpetra-style map
            full.replaceMap(map_->getImporter(block)->getSourceMap());      // temporarely switch to Xpetra GIDs

            // do the Export
            full.doExport(*rcpNonConstPartial, *(map_->getImporter(block)), Xpetra::INSERT);

            // switch back to original maps
            full.replaceMap(oldThyMapFull);                        // reset original map (Thyra GIDs)
            rcpNonConstPartial->replaceMap(oldThyMapPartial);      // change map back to original map
        }
        else
        {
            // Xpetra style numbering
            full.doExport(partial, *(map_->getImporter(block)), Xpetra::INSERT);
        }
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    InsertVector(const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>& partial, size_t block, MultiVector& full, bool bThyraMode) const
#else
    MapExtractor<Scalar, Node>::
    InsertVector(const Xpetra::MultiVector<Scalar,Node>& partial, size_t block, MultiVector& full, bool bThyraMode) const
#endif
    {
        XPETRA_TEST_FOR_EXCEPTION(block >= map_->getNumMaps(),
                                  std::out_of_range,
                                  "ExtractVector: Error, block = " << block << " is too big. The MapExtractor only contains " << map_->getNumMaps()
                                                                   << " partial blocks.");
        XPETRA_TEST_FOR_EXCEPTION(
          map_->getMap(block, false) == null, Xpetra::Exceptions::RuntimeError, "ExtractVector: map_->getmap(" << block << ",false) is null");
        XPETRA_TEST_FOR_EXCEPTION(map_->getThyraMode() == false && bThyraMode == true,
                                  Xpetra::Exceptions::RuntimeError,
                                  "MapExtractor::InsertVector: InsertVector in Thyra-style numbering only possible if MapExtractor has been created "
                                  "using Thyra-style numbered submaps.");
        if(bThyraMode)
        {
            // NOTE: the importer objects in the BlockedMap are always using Xpetra GIDs (or Thyra style Xpetra GIDs)
            // The source map corresponds to the full map (in Xpetra GIDs) starting with GIDs from zero. The GIDs are consecutive in Thyra mode
            // The target map is the partial map (in the corresponding Xpetra GIDs)

            // TODO can we skip the Export call in special cases (i.e. Src = Target map, same length, etc...)

            // store original GIDs (could be Thyra GIDs)
            RCP<const MultiVector> rcpPartial         = Teuchos::rcpFromRef(partial);
            RCP<MultiVector>       rcpNonConstPartial = Teuchos::rcp_const_cast<MultiVector>(rcpPartial);
            RCP<const Map>         oldThyMapPartial   = rcpNonConstPartial->getMap();      // temporarely store map of partial
            RCP<const Map>         oldThyMapFull      = full.getMap();                     // temporarely store map of full

            // check whether getMap(block,false) is identical to target map of importer
            XPETRA_TEST_FOR_EXCEPTION(map_->getMap(block, false)->isSameAs(*(map_->getImporter(block)->getTargetMap())) == false,
                                      Xpetra::Exceptions::RuntimeError,
                                      "MapExtractor::InsertVector: InsertVector in Thyra-style mode: Xpetra GIDs of partial vector are not identical "
                                      "to target Map of Importer. This should not be.");

            // XPETRA_TEST_FOR_EXCEPTION(full.getMap()->isSameAs(*(map_->getImporter(block)->getSourceMap()))==false,
            // Xpetra::Exceptions::RuntimeError,
            //           "MapExtractor::InsertVector: InsertVector in Thyra-style mode: Xpetra GIDs of full vector are not identical to source Map of
            //           Importer. This should not be.");

            rcpNonConstPartial->replaceMap(getMap(block, false));           // temporarely switch to xpetra-style map
            full.replaceMap(map_->getImporter(block)->getSourceMap());      // temporarely switch to Xpetra GIDs

            // do the Export
            full.doExport(*rcpNonConstPartial, *(map_->getImporter(block)), Xpetra::INSERT);

            // switch back to original maps
            full.replaceMap(oldThyMapFull);                        // reset original map (Thyra GIDs)
            rcpNonConstPartial->replaceMap(oldThyMapPartial);      // change map back to original map
        }
        else
        {
            // Xpetra style numbering
            full.doExport(partial, *(map_->getImporter(block)), Xpetra::INSERT);
        }
    }



#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    MapExtractor<Scalar, Node>::
#endif
    InsertVector(RCP<const Vector> partial, size_t block, RCP<Vector> full, bool bThyraMode) const
    {
        InsertVector(*partial, block, *full, bThyraMode);
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    MapExtractor<Scalar, Node>::
#endif
    InsertVector(RCP<Vector> partial, size_t block, RCP<Vector> full, bool bThyraMode) const
    {
        InsertVector(*partial, block, *full, bThyraMode);
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    InsertVector(RCP<const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > partial,
#else
    MapExtractor<Scalar, Node>::
    InsertVector(RCP<const Xpetra::MultiVector<Scalar,Node> > partial,
#endif
                 size_t block,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                 RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>> full,
#else
                 RCP<Xpetra::MultiVector<Scalar,Node>> full,
#endif
                 bool bThyraMode) const
    {
        RCP<BlockedMultiVector> bfull = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(full);
        if(bfull.is_null() == true)
            InsertVector(*partial, block, *full, bThyraMode);
        else
        {
            XPETRA_TEST_FOR_EXCEPTION(
              map_->getMap(block, false) == null, Xpetra::Exceptions::RuntimeError, "InsertVector: map_->getmap(" << block << ",false) is null");

            #if 0
            // WCMCLEN - ETI: MultiVector::setMultiVector() doesn't exist.
            // WCMCLEN - ETI: but BlockedMultiVector::setMultiVector() does... should this be using bfull.
            full->setMultiVector(block, partial, bThyraMode);
            #else
            throw std::runtime_error("Xpetra::MultiVector::setMultiVector() doesn't exist.");
            #endif
        }
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    InsertVector(RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > partial,
#else
    MapExtractor<Scalar, Node>::
    InsertVector(RCP<Xpetra::MultiVector<Scalar,Node> > partial,
#endif
                 size_t block,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                 RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>> full,
#else
                 RCP<Xpetra::MultiVector<Scalar,Node>> full,
#endif
                 bool bThyraMode) const
    {
        RCP<BlockedMultiVector> bfull = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(full);
        if(bfull.is_null() == true)
            InsertVector(*partial, block, *full, bThyraMode);
        else
        {
            XPETRA_TEST_FOR_EXCEPTION(
              map_->getMap(block, false) == null, Xpetra::Exceptions::RuntimeError, "InsertVector: map_->getmap(" << block << ",false) is null");

            bfull->setMultiVector(block, partial, bThyraMode);
        }
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    InsertVector(RCP<const Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>>     partial,
#else
    MapExtractor<Scalar, Node>::
    InsertVector(RCP<const Xpetra::MultiVector<Scalar,Node>>     partial,
#endif
                 size_t                                                                     block,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                 RCP<Xpetra::BlockedMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> full,
#else
                 RCP<Xpetra::BlockedMultiVector<Scalar, Node>> full,
#endif
                 bool                                                                       bThyraMode) const
    {
        XPETRA_TEST_FOR_EXCEPTION(
          map_->getMap(block, false) == null, Xpetra::Exceptions::RuntimeError, "InsertVector: map_->getmap(" << block << ",false) is null");

        full->setMultiVector(block, partial, bThyraMode);
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    InsertVector(RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>        partial,
#else
    MapExtractor<Scalar, Node>::
    InsertVector(RCP<Xpetra::MultiVector<Scalar, Node>>        partial,
#endif
                 size_t                                                                     block,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                 RCP<Xpetra::BlockedMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> full,
#else
                 RCP<Xpetra::BlockedMultiVector<Scalar, Node>> full,
#endif
                 bool                                                                       bThyraMode ) const
    {
        XPETRA_TEST_FOR_EXCEPTION(
          map_->getMap(block, false) == null, Xpetra::Exceptions::RuntimeError, "InsertVector: map_->getmap(" << block << ",false) is null");
        full->setMultiVector(block, partial, bThyraMode);
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    RCP<Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    template <class Scalar, class Node>
    RCP<Xpetra::Vector<Scalar, Node>>
    MapExtractor<Scalar, Node>::
#endif
    getVector(size_t i, bool bThyraMode, bool bZero) const
    {
        XPETRA_TEST_FOR_EXCEPTION(map_->getThyraMode() == false && bThyraMode == true,
                                  Xpetra::Exceptions::RuntimeError,
                                  "MapExtractor::getVector: getVector in Thyra-style numbering only possible if MapExtractor has been created using "
                                  "Thyra-style numbered submaps.");
        // TODO check whether this can return a blocked multivector
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(getMap(i, bThyraMode), bZero);
#else
        return Xpetra::VectorFactory<Scalar,Node>::Build(getMap(i, bThyraMode), bZero);
#endif
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>>
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    template <class Scalar, class Node>
    RCP<Xpetra::MultiVector<Scalar,Node>>
    MapExtractor<Scalar, Node>::
#endif
    getVector(size_t i, size_t numvec, bool bThyraMode, bool bZero) const
    {
        XPETRA_TEST_FOR_EXCEPTION(map_->getThyraMode() == false && bThyraMode == true,
                                  Xpetra::Exceptions::RuntimeError,
                                  "MapExtractor::getVector: getVector in Thyra-style numbering only possible if MapExtractor has been created using "
                                  "Thyra-style numbered submaps.");
        // TODO check whether this can return a blocked multivector
        return MultiVectorFactory::Build(getMap(i, bThyraMode), numvec, bZero);
    }

    /// returns true, if sub maps are stored in Thyra-style numbering
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    bool
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    MapExtractor<Scalar, Node>::
#endif
    getThyraMode() const
    {
        return map_->getThyraMode();
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    size_t
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    MapExtractor<Scalar, Node>::
#endif
    NumMaps() const
    {
        return map_->getNumMaps();
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    const RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>>
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    template <class Scalar, class Node>
    const RCP<const Xpetra::Map<Node>>
    MapExtractor<Scalar, Node>::
#endif
    getMap(size_t i, bool bThyraMode) const
    {
        return map_->getMap(i, bThyraMode);
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    const RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>>
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    template <class Scalar, class Node>
    const RCP<const Xpetra::Map<Node>>
    MapExtractor<Scalar, Node>::
#endif
    getMap() const
    {
        return map_;
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    const RCP<const Xpetra::BlockedMap<LocalOrdinal,GlobalOrdinal,Node>>
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    template <class Scalar, class Node>
    const RCP<const Xpetra::BlockedMap<Node>>
    MapExtractor<Scalar, Node>::
#endif
    getBlockedMap() const
    {
        return map_;
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    const RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>>
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    template <class Scalar, class Node>
    const RCP<const Xpetra::Map<Node>>
    MapExtractor<Scalar, Node>::
#endif
    getFullMap() const
    {
        return map_->getFullMap();
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
#else
    template <class Scalar, class Node>
#endif
    size_t
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    MapExtractor<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
#else
    MapExtractor<Scalar, Node>::
#endif
    getMapIndexForGID(GlobalOrdinal gid) const
    {
        return map_->getMapIndexForGID(gid);
    }


} // namespace Xpetra

#endif /* XPETRA_MAPEXTRACTOR_DEF_HPP_ */
