// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
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
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER

#ifndef TPETRA_DIRECTORY_HPP
#define TPETRA_DIRECTORY_HPP

#include "Tpetra_Distributor.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_DirectoryImpl.hpp"
#include "Tpetra_Directory_decl.hpp"

namespace Tpetra {

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class LO, class GO, class NT>
  Directory<LO, GO, NT>::Directory () :
#else
  template<class NT>
  Directory<NT>::Directory () :
#endif
    impl_ (NULL)
  {}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class LO, class GO, class NT>
  Directory<LO, GO, NT>::~Directory () {
#else
  template<class NT>
  Directory<NT>::~Directory () {
#endif
    if (impl_ != NULL) {
      delete impl_;
      impl_ = NULL;
    }
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class LO, class GO, class NT>
#else
  template<class NT>
#endif
  bool
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  Directory<LO, GO, NT>::initialized () const {
#else
  Directory<NT>::initialized () const {
#endif
    return impl_ != NULL;
  }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class LO, class GO, class NT>
#else
  template<class NT>
#endif
  void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  Directory<LO, GO, NT>::
  initialize (const Map<LO, GO, NT>& map,
#else
  Directory<NT>::
  initialize (const Map<NT>& map,
#endif
              const Tpetra::Details::TieBreak<LO,GO>& tieBreak)
  {
    if (initialized ()) {
      TEUCHOS_TEST_FOR_EXCEPTION(
        impl_ == NULL, std::logic_error, "Tpetra::Directory::initialize: "
        "The Directory claims that it has been initialized, "
        "but its implementation object has not yet been created.  "
        "Please report this bug to the Tpetra developers.");
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(
        impl_ != NULL, std::logic_error, "Tpetra::Directory::initialize: "
        "Directory implementation has already been initialized, "
        "but initialized() returns false.  "
        "Please report this bug to the Tpetra developers.");

      // Create an implementation object of the appropriate type,
      // depending on whether the Map is distributed or replicated,
      // and contiguous or noncontiguous.
      //
      // mfh 06 Apr 2014: When a distributed noncontiguous Directory
      // takes a TieBreak, all the entries (local indices and process
      // ranks) owned by the Directory on the calling process pass
      // through the TieBreak object.  This may have side effects,
      // such as the TieBreak object remembering whether there were
      // any duplicates on the calling process.  We want to extend use
      // of a TieBreak object to other kinds of Directories.  For a
      // distributed contiguous Directory, the calling process owns
      // all of the (PID,LID) pairs in the input Map.  For a locally
      // replicated contiguous Directory, Process 0 owns all of the
      // (PID,LID) pairs in the input Map.
      //
      // It may seem silly to pass in a TieBreak when there are no
      // ties to break.  However, the TieBreak object gets to see all
      // (PID,LID) pairs that the Directory owns on the calling
      // process, and interface of TieBreak allows side effects.
      // Users may wish to exploit them regardless of the kind of Map
      // they pass in.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const ::Tpetra::Details::Directory<LO, GO, NT>* dir = NULL;
#else
      const ::Tpetra::Details::Directory<NT>* dir = NULL;
#endif
      bool usedTieBreak = false;
      if (map.isDistributed ()) {
        if (map.isUniform ()) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
          dir = new ::Tpetra::Details::ContiguousUniformDirectory<LO, GO, NT> (map);
#else
          dir = new ::Tpetra::Details::ContiguousUniformDirectory<NT> (map);
#endif
        }
        else if (map.isContiguous ()) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
          dir = new ::Tpetra::Details::DistributedContiguousDirectory<LO, GO, NT> (map);
#else
          dir = new ::Tpetra::Details::DistributedContiguousDirectory<NT> (map);
#endif
        }
        else {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
          dir = new ::Tpetra::Details::DistributedNoncontiguousDirectory<LO, GO, NT> (map, tieBreak);
#else
          dir = new ::Tpetra::Details::DistributedNoncontiguousDirectory<NT> (map, tieBreak);
#endif
          usedTieBreak = true;
        }
      }
      else {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        dir = new ::Tpetra::Details::ReplicatedDirectory<LO, GO, NT> (map);
#else
        dir = new ::Tpetra::Details::ReplicatedDirectory<NT> (map);
#endif

        if (tieBreak.mayHaveSideEffects () && map.getNodeNumElements () != 0) {
          // We need the second clause in the above test because Map's
          // interface provides an inclusive range of local indices.
          const int myRank = map.getComm ()->getRank ();
          // In a replicated Directory, Process 0 owns all the
          // Directory's entries.  This is an arbitrary assignment; any
          // one process would do.
          if (myRank == 0) {
            std::vector<std::pair<int, LO> > pidLidList (1);
            const LO minLocInd = map.getMinLocalIndex ();
            const LO maxLocInd = map.getMaxLocalIndex ();
            for (LO locInd = minLocInd; locInd <= maxLocInd; ++locInd) {
              pidLidList[0] = std::make_pair (myRank, locInd);
              const GO globInd = map.getGlobalElement (locInd);
              // We don't care about the return value; we just want to
              // invoke the side effects.
              (void) tieBreak.selectedIndex (globInd, pidLidList);
            }
          }
        }
        usedTieBreak = true;
      } // done with all different Map cases

      // If we haven't already used the TieBreak object, use it now.
      // This code appears twice because ReplicatedDirectory is a
      // special case: we already know what gets replicated.
      if (! usedTieBreak && tieBreak.mayHaveSideEffects () &&
          map.getNodeNumElements () != 0) {
        // We need the third clause in the above test because Map's
        // interface provides an inclusive range of local indices.
        std::vector<std::pair<int, LO> > pidLidList (1);
        const LO minLocInd = map.getMinLocalIndex ();
        const LO maxLocInd = map.getMaxLocalIndex ();
        const int myRank = map.getComm ()->getRank ();
        for (LO locInd = minLocInd; locInd <= maxLocInd; ++locInd) {
          pidLidList[0] = std::make_pair (myRank, locInd);
          const GO globInd = map.getGlobalElement (locInd);
          // We don't care about the return value; we just want to
          // invoke the side effects.
          (void) tieBreak.selectedIndex (globInd, pidLidList);
        }
      }

      impl_ = dir;
    }
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class LO, class GO, class NT>
#else
  template<class NT>
#endif
  void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  Directory<LO, GO, NT>::initialize (const Map<LO, GO, NT>& map)
#else
  Directory<NT>::initialize (const Map<NT>& map)
#endif
  {
    if (initialized ()) {
      TEUCHOS_TEST_FOR_EXCEPTION(
        impl_ == NULL, std::logic_error, "Tpetra::Directory::initialize: "
        "The Directory claims that it has been initialized, "
        "but its implementation object has not yet been created.  "
        "Please report this bug to the Tpetra developers.");
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(
        impl_ != NULL, std::logic_error, "Tpetra::Directory::initialize: "
        "Directory implementation has already been initialized, "
        "but initialized() returns false.  "
        "Please report this bug to the Tpetra developers.");

      // Create an implementation object of the appropriate type,
      // depending on whether the Map is distributed or replicated,
      // and contiguous or noncontiguous.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const ::Tpetra::Details::Directory<LO, GO, NT>* dir = NULL;
#else
      const ::Tpetra::Details::Directory<NT>* dir = NULL;
#endif
      if (map.isDistributed ()) {
        if (map.isUniform ()) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
          dir = new ::Tpetra::Details::ContiguousUniformDirectory<LO, GO, NT> (map);
#else
          dir = new ::Tpetra::Details::ContiguousUniformDirectory<NT> (map);
#endif
        }
        else if (map.isContiguous ()) {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
          dir = new ::Tpetra::Details::DistributedContiguousDirectory<LO, GO, NT> (map);
#else
          dir = new ::Tpetra::Details::DistributedContiguousDirectory<NT> (map);
#endif
        }
        else {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
          dir = new ::Tpetra::Details::DistributedNoncontiguousDirectory<LO, GO, NT> (map);
#else
          dir = new ::Tpetra::Details::DistributedNoncontiguousDirectory<NT> (map);
#endif
        }
      }
      else {
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        dir = new ::Tpetra::Details::ReplicatedDirectory<LO, GO, NT> (map);
#else
        dir = new ::Tpetra::Details::ReplicatedDirectory<NT> (map);
#endif
      }
      TEUCHOS_TEST_FOR_EXCEPTION(
        dir == NULL, std::logic_error, "Tpetra::Directory::initialize: "
        "Failed to create Directory implementation.  "
        "Please report this bug to the Tpetra developers.");
      impl_ = dir;
    }
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class LO, class GO, class NT>
#else
  template<class NT>
#endif
  LookupStatus
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  Directory<LO, GO, NT>::
  getDirectoryEntries (const Map<LO, GO, NT>& map,
#else
  Directory<NT>::
  getDirectoryEntries (const Map<NT>& map,
#endif
                       const Teuchos::ArrayView<const GO>& globalIDs,
                       const Teuchos::ArrayView<int>& nodeIDs) const
  {
    if (! initialized ()) {
      // This const_cast is super wrong, but "mutable" is also a lie,
      // and Map's interface needs this method to be marked const for
      // some reason.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const_cast<Directory<LO, GO, NT>* > (this)->initialize (map);
#else
      const_cast<Directory<NT>* > (this)->initialize (map);
#endif
    }
    const bool computeLIDs = false;
    return impl_->getEntries (map, globalIDs, nodeIDs, Teuchos::null, computeLIDs);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class LO, class GO, class NT>
#else
  template<class NT>
#endif
  LookupStatus
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  Directory<LO, GO, NT>::
  getDirectoryEntries (const Map<LO, GO, NT>& map,
#else
  Directory<NT>::
  getDirectoryEntries (const Map<NT>& map,
#endif
                       const Teuchos::ArrayView<const GO>& globalIDs,
                       const Teuchos::ArrayView<int>& nodeIDs,
                       const Teuchos::ArrayView<LO>& localIDs) const
  {
    if (! initialized ()) {
      // This const_cast is super wrong, but "mutable" is also a lie,
      // and Map's interface needs this method to be marked const for
      // some reason.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const_cast<Directory<LO, GO, NT>* > (this)->initialize (map);
#else
      const_cast<Directory<NT>* > (this)->initialize (map);
#endif
    }
    const bool computeLIDs = true;
    return impl_->getEntries (map, globalIDs, nodeIDs, localIDs, computeLIDs);
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class LO, class GO, class NT>
  bool Directory<LO, GO, NT>::isOneToOne (const Map<LO, GO, NT>& map) const {
#else
  template<class NT>
  bool Directory<NT>::isOneToOne (const Map<NT>& map) const {
#endif
    if (! initialized ()) {
      // This const_cast is super wrong, but "mutable" is also a lie,
      // and Map's interface needs this method to be marked const for
      // some reason.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      const_cast<Directory<LO, GO, NT>* > (this)->initialize (map);
#else
      const_cast<Directory<NT>* > (this)->initialize (map);
#endif
    }
    return impl_->isOneToOne (* (map.getComm ()));
  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  template<class LO, class GO, class NT>
#else
  template<class NT>
#endif
  std::string
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  Directory<LO, GO, NT>::description () const
#else
  Directory<NT>::description () const
#endif
  {
    using Teuchos::TypeNameTraits;

    std::ostringstream os;
    os << "Directory"
       << "<" << TypeNameTraits<LO>::name ()
       << ", " << TypeNameTraits<GO>::name ()
       << ", " << TypeNameTraits<NT>::name () << ">";
    return os.str ();
  }

} // namespace Tpetra

//
// Explicit instantiation macro
//
// Must be expanded from within the Tpetra namespace!
//

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
#define TPETRA_DIRECTORY_INSTANT(LO,GO,NODE) \
  template class Directory< LO , GO , NODE >;
#else
#define TPETRA_DIRECTORY_INSTANT(NODE) \
  template class Directory<NODE >;
#endif

#endif // TPETRA_DIRECTORY_HPP
