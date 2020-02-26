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
#ifndef XPETRA_TPETRAMAP_DEF_HPP
#define XPETRA_TPETRAMAP_DEF_HPP

#include "Xpetra_TpetraMap_decl.hpp"



namespace Xpetra {


//! @name Constructor/Destructor Methods
//@{




#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::
#else
    template<class Node>
    TpetraMap<Node>::
#endif
    TpetraMap (global_size_t numGlobalElements,
               GlobalOrdinal indexBase,
               const Teuchos::RCP< const Teuchos::Comm< int > > &comm,
               LocalGlobal lg)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      : map_ (Teuchos::rcp (new Tpetra::Map< LocalOrdinal, GlobalOrdinal, Node > (numGlobalElements,
#else
      : map_ (Teuchos::rcp (new Tpetra::Map<Node > (numGlobalElements,
#endif
                                                                                  indexBase, comm,
                                                                                  toTpetra(lg))))
    {}


    //! Constructor with a user-defined contiguous distribution.


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::
#else
    template<class Node>
    TpetraMap<Node>::
#endif
    TpetraMap (global_size_t numGlobalElements,
               size_t numLocalElements,
               GlobalOrdinal indexBase,
               const Teuchos::RCP< const Teuchos::Comm< int > > &comm)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      : map_ (Teuchos::rcp (new Tpetra::Map< LocalOrdinal, GlobalOrdinal, Node > (numGlobalElements,
#else
      : map_ (Teuchos::rcp (new Tpetra::Map<Node > (numGlobalElements,
#endif
                                                                                  numLocalElements,
                                                                                  indexBase, comm)))
    {}


    //! Constructor with user-defined arbitrary (possibly noncontiguous) distribution.


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::
#else
    template<class Node>
    TpetraMap<Node>::
#endif
    TpetraMap (global_size_t numGlobalElements,
               const Teuchos::ArrayView< const GlobalOrdinal > &elementList,
               GlobalOrdinal indexBase,
               const Teuchos::RCP< const Teuchos::Comm< int > > &comm)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      : map_(Teuchos::rcp(new Tpetra::Map< LocalOrdinal, GlobalOrdinal, Node >(numGlobalElements,
#else
      : map_(Teuchos::rcp(new Tpetra::Map<Node >(numGlobalElements,
#endif
                                                                               elementList,
                                                                               indexBase,
                                                                               comm)))
    {}


#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef HAVE_XPETRA_TPETRA

    //! Constructor with user-defined arbitrary (possibly noncontiguous) distribution passed as a Kokkos::View.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template<class LocalOrdinal, class GlobalOrdinal, class Node>
    TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::
#else
    template<class Node>
    TpetraMap<Node>::
#endif
    TpetraMap (global_size_t numGlobalElements,
               const Kokkos::View<const GlobalOrdinal*, typename Node::device_type>& indexList,
               GlobalOrdinal indexBase,
               const Teuchos::RCP< const Teuchos::Comm< int > > &comm)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
      : map_(Teuchos::rcp(new Tpetra::Map< LocalOrdinal, GlobalOrdinal, Node >(numGlobalElements,
#else
      : map_(Teuchos::rcp(new Tpetra::Map<Node >(numGlobalElements,
#endif
                                                                               indexList,
                                                                               indexBase,
                                                                               comm)))
    {}
#endif
#endif

//! Destructor.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::~TpetraMap() 
#else
template<class Node>
TpetraMap<Node>::~TpetraMap() 
#endif
{  }

    //@}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
global_size_t TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::getGlobalNumElements() const
#else
template<class Node>
global_size_t TpetraMap<Node>::getGlobalNumElements() const
#endif
{ XPETRA_MONITOR("TpetraMap::getGlobalNumElements"); return map_->getGlobalNumElements(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
size_t TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::getNodeNumElements() const
#else
template<class Node>
size_t TpetraMap<Node>::getNodeNumElements() const
#endif
{ XPETRA_MONITOR("TpetraMap::getNodeNumElements"); return map_->getNodeNumElements(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
GlobalOrdinal TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::getIndexBase() const
#else
template<class Node>
GlobalOrdinal TpetraMap<Node>::getIndexBase() const
#endif
{ XPETRA_MONITOR("TpetraMap::getIndexBase"); return map_->getIndexBase(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
LocalOrdinal TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::getMinLocalIndex() const
#else
template<class Node>
LocalOrdinal TpetraMap<Node>::getMinLocalIndex() const
#endif
{ XPETRA_MONITOR("TpetraMap::getMinLocalIndex"); return map_->getMinLocalIndex(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
LocalOrdinal TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::getMaxLocalIndex() const
#else
template<class Node>
LocalOrdinal TpetraMap<Node>::getMaxLocalIndex() const
#endif
{ XPETRA_MONITOR("TpetraMap::getMaxLocalIndex"); return map_->getMaxLocalIndex(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
GlobalOrdinal TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::getMinGlobalIndex() const
#else
template<class Node>
GlobalOrdinal TpetraMap<Node>::getMinGlobalIndex() const
#endif
{ XPETRA_MONITOR("TpetraMap::getMinGlobalIndex"); return map_->getMinGlobalIndex(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
GlobalOrdinal TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::getMaxGlobalIndex() const
#else
template<class Node>
GlobalOrdinal TpetraMap<Node>::getMaxGlobalIndex() const
#endif
{ XPETRA_MONITOR("TpetraMap::getMaxGlobalIndex"); return map_->getMaxGlobalIndex(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
GlobalOrdinal TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::getMinAllGlobalIndex() const
#else
template<class Node>
GlobalOrdinal TpetraMap<Node>::getMinAllGlobalIndex() const
#endif
{ XPETRA_MONITOR("TpetraMap::getMinAllGlobalIndex"); return map_->getMinAllGlobalIndex(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
GlobalOrdinal TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::getMaxAllGlobalIndex() const
#else
template<class Node>
GlobalOrdinal TpetraMap<Node>::getMaxAllGlobalIndex() const
#endif
{ XPETRA_MONITOR("TpetraMap::getMaxAllGlobalIndex"); return map_->getMaxAllGlobalIndex(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
LocalOrdinal TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::getLocalElement(GlobalOrdinal globalIndex) const
#else
template<class Node>
LocalOrdinal TpetraMap<Node>::getLocalElement(GlobalOrdinal globalIndex) const
#endif
{ XPETRA_MONITOR("TpetraMap::getLocalElement"); return map_->getLocalElement(globalIndex); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
GlobalOrdinal TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::getGlobalElement(LocalOrdinal localIndex) const
#else
template<class Node>
GlobalOrdinal TpetraMap<Node>::getGlobalElement(LocalOrdinal localIndex) const
#endif
{ XPETRA_MONITOR("TpetraMap::getGlobalElement"); return map_->getGlobalElement(localIndex); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
LookupStatus TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::getRemoteIndexList(const Teuchos::ArrayView< const GlobalOrdinal > &GIDList, const Teuchos::ArrayView< int > &nodeIDList, const Teuchos::ArrayView< LocalOrdinal > &LIDList) const
#else
template<class Node>
LookupStatus TpetraMap<Node>::getRemoteIndexList(const Teuchos::ArrayView< const GlobalOrdinal > &GIDList, const Teuchos::ArrayView< int > &nodeIDList, const Teuchos::ArrayView< LocalOrdinal > &LIDList) const
#endif
{ XPETRA_MONITOR("TpetraMap::getRemoteIndexList"); return toXpetra(map_->getRemoteIndexList(GIDList, nodeIDList, LIDList)); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
LookupStatus TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::getRemoteIndexList(const Teuchos::ArrayView< const GlobalOrdinal > &GIDList, const Teuchos::ArrayView< int > &nodeIDList) const
#else
template<class Node>
LookupStatus TpetraMap<Node>::getRemoteIndexList(const Teuchos::ArrayView< const GlobalOrdinal > &GIDList, const Teuchos::ArrayView< int > &nodeIDList) const
#endif
{ XPETRA_MONITOR("TpetraMap::getRemoteIndexList"); return toXpetra(map_->getRemoteIndexList(GIDList, nodeIDList)); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::ArrayView< const GlobalOrdinal > TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::getNodeElementList() const
#else
template<class Node>
Teuchos::ArrayView< const GlobalOrdinal > TpetraMap<Node>::getNodeElementList() const
#endif
{ XPETRA_MONITOR("TpetraMap::getNodeElementList"); return map_->getNodeElementList(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
bool TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::isNodeLocalElement(LocalOrdinal localIndex) const
#else
template<class Node>
bool TpetraMap<Node>::isNodeLocalElement(LocalOrdinal localIndex) const
#endif
{ XPETRA_MONITOR("TpetraMap::isNodeLocalElement"); return map_->isNodeLocalElement(localIndex); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
bool TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::isNodeGlobalElement(GlobalOrdinal globalIndex) const
#else
template<class Node>
bool TpetraMap<Node>::isNodeGlobalElement(GlobalOrdinal globalIndex) const
#endif
{ XPETRA_MONITOR("TpetraMap::isNodeGlobalElement"); return map_->isNodeGlobalElement(globalIndex); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
bool TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::isContiguous() const
#else
template<class Node>
bool TpetraMap<Node>::isContiguous() const
#endif
{ XPETRA_MONITOR("TpetraMap::isContiguous"); return map_->isContiguous(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
bool TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::isDistributed() const
#else
template<class Node>
bool TpetraMap<Node>::isDistributed() const
#endif
{ XPETRA_MONITOR("TpetraMap::isDistributed"); return map_->isDistributed(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
bool TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::isCompatible(const Map< LocalOrdinal, GlobalOrdinal, Node > &map) const
#else
template<class Node>
bool TpetraMap<Node>::isCompatible(const Map<Node > &map) const
#endif
{ XPETRA_MONITOR("TpetraMap::isCompatible"); return map_->isCompatible(toTpetra(map)); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
bool TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::isSameAs(const Map< LocalOrdinal, GlobalOrdinal, Node > &map) const
#else
template<class Node>
bool TpetraMap<Node>::isSameAs(const Map<Node > &map) const
#endif
{ XPETRA_MONITOR("TpetraMap::isSameAs"); return map_->isSameAs(toTpetra(map)); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP< const Teuchos::Comm< int > >  TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::getComm() const
#else
template<class Node>
Teuchos::RCP< const Teuchos::Comm< int > >  TpetraMap<Node>::getComm() const
#endif
{ XPETRA_MONITOR("TpetraMap::getComm"); return map_->getComm(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
std::string TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::description() const
#else
template<class Node>
std::string TpetraMap<Node>::description() const
#endif
{ XPETRA_MONITOR("TpetraMap::description"); return map_->description(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel) const
#else
template<class Node>
void TpetraMap<Node>::describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel) const
#endif
{ XPETRA_MONITOR("TpetraMap::describe"); map_->describe(out, verbLevel); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::removeEmptyProcesses () const
#else
template<class Node>
RCP<const Map<Node> > TpetraMap<Node>::removeEmptyProcesses () const
#endif
{
    return toXpetra(map_->removeEmptyProcesses());
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::replaceCommWithSubset (const Teuchos::RCP<const Teuchos::Comm<int> >& newComm) const
#else
template<class Node>
RCP<const Map<Node> > TpetraMap<Node>::replaceCommWithSubset (const Teuchos::RCP<const Teuchos::Comm<int> >& newComm) const
#endif
{
    return toXpetra(map_->replaceCommWithSubset(newComm));
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::TpetraMap(const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node > > &map)
#else
template<class Node>
TpetraMap<Node>::TpetraMap(const Teuchos::RCP<const Tpetra::Map<Node > > &map)
#endif
: map_(map) { }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
UnderlyingLib TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::lib() const { return UseTpetra; }
#else
template<class Node>
UnderlyingLib TpetraMap<Node>::lib() const { return UseTpetra; }
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
RCP< const Tpetra::Map< LocalOrdinal, GlobalOrdinal, Node > > TpetraMap<LocalOrdinal,GlobalOrdinal,Node>::getTpetra_Map() const
#else
template<class Node>
RCP< const Tpetra::Map<Node > > TpetraMap<Node>::getTpetra_Map() const
#endif
{ return map_; }


#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef HAVE_XPETRA_TPETRA

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
typename Map<LocalOrdinal, GlobalOrdinal, Node>::local_map_type TpetraMap<LocalOrdinal, GlobalOrdinal, Node>::getLocalMap () const
#else
template<class Node>
typename Map<Node>::local_map_type TpetraMap<Node>::getLocalMap () const
#endif
{
    return map_->getLocalMap();
}
#endif
#endif


#ifdef HAVE_XPETRA_EPETRA

#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))

  // specialization for Tpetra Map on EpetraNode and GO=int
  template <>
  class TpetraMap<int, int, EpetraNode>
    : public virtual Map<int,int,EpetraNode> {

  public:
    typedef int GlobalOrdinal;
    typedef int LocalOrdinal;
    typedef EpetraNode Node;

    //! @name Constructors and destructor
    //@{


    //! Constructor with Tpetra-defined contiguous uniform distribution.


    TpetraMap (global_size_t numGlobalElements,
               GlobalOrdinal indexBase,
               const Teuchos::RCP< const Teuchos::Comm< int > > &comm,
               LocalGlobal lg=GloballyDistributed) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMap<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMap<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }


    //! Constructor with a user-defined contiguous distribution.


    TpetraMap (global_size_t numGlobalElements,
               size_t numLocalElements,
               GlobalOrdinal indexBase,
               const Teuchos::RCP< const Teuchos::Comm< int > > &comm) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMap<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMap<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }


    //! Constructor with user-defined arbitrary (possibly noncontiguous) distribution.


    TpetraMap (global_size_t numGlobalElements,
               const Teuchos::ArrayView< const GlobalOrdinal > &elementList,
               GlobalOrdinal indexBase,
               const Teuchos::RCP< const Teuchos::Comm< int > > &comm) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMap<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMap<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }


    //! Destructor.
    ~TpetraMap() {  }

    //@}

    //! @name Attributes
    //@{

    //! The number of elements in this Map.
    global_size_t getGlobalNumElements() const { return 0; }

    //! The number of elements belonging to the calling node.
    size_t getNodeNumElements() const { return 0; }

    //! The index base for this Map.
    GlobalOrdinal getIndexBase() const { return 0; }

    //! The minimum local index.
    LocalOrdinal getMinLocalIndex() const { return 0; }

    //! The maximum local index on the calling process.
    LocalOrdinal getMaxLocalIndex() const { return 0; }

    //! The minimum global index owned by the calling process.
    GlobalOrdinal getMinGlobalIndex() const { return 0; }

    //! The maximum global index owned by the calling process.
    GlobalOrdinal getMaxGlobalIndex() const { return 0; }

    //! The minimum global index over all processes in the communicator.
    GlobalOrdinal getMinAllGlobalIndex() const { return 0; }

    //! The maximum global index over all processes in the communicator.
    GlobalOrdinal getMaxAllGlobalIndex() const { return 0; }

    //! The local index corresponding to the given global index.
    LocalOrdinal getLocalElement(GlobalOrdinal globalIndex) const { return 0; }

    //! The global index corresponding to the given local index.
    GlobalOrdinal getGlobalElement(LocalOrdinal localIndex) const { return 0; }

    //! Return the process IDs and corresponding local IDs for the given global IDs.
    LookupStatus getRemoteIndexList(const Teuchos::ArrayView< const GlobalOrdinal > &GIDList, const Teuchos::ArrayView< int > &nodeIDList, const Teuchos::ArrayView< LocalOrdinal > &LIDList) const { return Xpetra::IDNotPresent; }

    //! Return the process IDs for the given global IDs.
    LookupStatus getRemoteIndexList(const Teuchos::ArrayView< const GlobalOrdinal > &GIDList, const Teuchos::ArrayView< int > &nodeIDList) const { return Xpetra::IDNotPresent; }

    //! Return a view of the global indices owned by this node.
    Teuchos::ArrayView< const GlobalOrdinal > getNodeElementList() const { return Teuchos::ArrayView<const GlobalOrdinal>(); }

    //@}

    //! @name Boolean tests
    //@{

    //! True if the local index is valid for this Map on this node, else false.
    bool isNodeLocalElement(LocalOrdinal localIndex) const { return false; }

    //! True if the global index is found in this Map on this node, else false.
    bool isNodeGlobalElement(GlobalOrdinal globalIndex) const { return false; }

    //! True if this Map is distributed contiguously, else false.
    bool isContiguous() const { return false; }

    //! Whether this Map is globally distributed or locally replicated.
    bool isDistributed() const { return false; }

    //! True if and only if map is compatible with this Map.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    bool isCompatible(const Map< LocalOrdinal, GlobalOrdinal, Node > &map) const { return false; }
#else
    bool isCompatible(const Map<Node > &map) const { return false; }
#endif

    //! True if and only if map is identical to this Map.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    bool isSameAs(const Map< LocalOrdinal, GlobalOrdinal, Node > &map) const { return false; }
#else
    bool isSameAs(const Map<Node > &map) const { return false; }
#endif

    //@}

    //! @name
    //@{

    //! Get this Map's Comm object.
    Teuchos::RCP< const Teuchos::Comm< int > >  getComm() const { return Teuchos::null; }


    //@}

    //! @name
    //@{

    //! Return a simple one-line description of this object.
    std::string description() const { return std::string(""); }

    //! Print this object with the given verbosity level to the given FancyOStream.
    void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel=Teuchos::Describable::verbLevel_default) const { }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > removeEmptyProcesses () const { return Teuchos::null; }
    RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > replaceCommWithSubset (const Teuchos::RCP<const Teuchos::Comm<int> >& newComm) const { return Teuchos::null; }
#else
    RCP<const Map<Node> > removeEmptyProcesses () const { return Teuchos::null; }
    RCP<const Map<Node> > replaceCommWithSubset (const Teuchos::RCP<const Teuchos::Comm<int> >& newComm) const { return Teuchos::null; }
#endif

//@}

    //! @name Xpetra specific
    //@{

    //! TpetraMap constructor to wrap a Tpetra::Map object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraMap(const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node > > &map) {
#else
    TpetraMap(const Teuchos::RCP<const Tpetra::Map<Node > > &map) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMap<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMap<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Get the library used by this object (Tpetra or Epetra?)
    UnderlyingLib lib() const { return UseTpetra; }

    //! Get the underlying Tpetra map
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Tpetra::Map< LocalOrdinal, GlobalOrdinal, Node > > getTpetra_Map() const { return Teuchos::null; }
#else
    RCP< const Tpetra::Map<Node > > getTpetra_Map() const { return Teuchos::null; }
#endif

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef HAVE_XPETRA_TPETRA
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    using local_map_type = typename Map<LocalOrdinal, GlobalOrdinal, Node>::local_map_type;
#else
    using local_map_type = typename Map<Node>::local_map_type;
#endif
    /// \brief Get the local Map for Kokkos kernels.
    local_map_type getLocalMap () const {
      return local_map_type();
    }
#endif
#endif

    //@}

  }; // TpetraMap class (specialization for GO=int and NO=EpetraNode)
#endif

#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_LONG_LONG))))
  // specialization for Tpetra Map on EpetraNode and GO=int
  template <>
  class TpetraMap<int, long long, EpetraNode>
    : public virtual Map<int,long long,EpetraNode> {

  public:
    typedef long long GlobalOrdinal;
    typedef int LocalOrdinal;
    typedef EpetraNode Node;

    //! @name Constructors and destructor
    //@{

    //! Constructor with Tpetra-defined contiguous uniform distribution.
    TpetraMap (global_size_t numGlobalElements,
               GlobalOrdinal indexBase,
               const Teuchos::RCP< const Teuchos::Comm< int > > &comm,
               LocalGlobal lg=GloballyDistributed) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMap<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMap<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

    //! Constructor with a user-defined contiguous distribution.
    TpetraMap (global_size_t numGlobalElements,
               size_t numLocalElements,
               GlobalOrdinal indexBase,
               const Teuchos::RCP< const Teuchos::Comm< int > > &comm) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMap<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMap<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

    //! Constructor with user-defined arbitrary (possibly noncontiguous) distribution.
    TpetraMap (global_size_t numGlobalElements,
               const Teuchos::ArrayView< const GlobalOrdinal > &elementList,
               GlobalOrdinal indexBase,
               const Teuchos::RCP< const Teuchos::Comm< int > > &comm) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMap<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMap<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

    //! Destructor.
    ~TpetraMap() {  }

    //@}

    //! @name Attributes
    //@{

    //! The number of elements in this Map.
    global_size_t getGlobalNumElements() const { return 0; }

    //! The number of elements belonging to the calling node.
    size_t getNodeNumElements() const { return 0; }

    //! The index base for this Map.
    GlobalOrdinal getIndexBase() const { return 0; }

    //! The minimum local index.
    LocalOrdinal getMinLocalIndex() const { return 0; }

    //! The maximum local index on the calling process.
    LocalOrdinal getMaxLocalIndex() const { return 0; }

    //! The minimum global index owned by the calling process.
    GlobalOrdinal getMinGlobalIndex() const { return 0; }

    //! The maximum global index owned by the calling process.
    GlobalOrdinal getMaxGlobalIndex() const { return 0; }

    //! The minimum global index over all processes in the communicator.
    GlobalOrdinal getMinAllGlobalIndex() const { return 0; }

    //! The maximum global index over all processes in the communicator.
    GlobalOrdinal getMaxAllGlobalIndex() const { return 0; }

    //! The local index corresponding to the given global index.
    LocalOrdinal getLocalElement(GlobalOrdinal globalIndex) const { return 0; }

    //! The global index corresponding to the given local index.
    GlobalOrdinal getGlobalElement(LocalOrdinal localIndex) const { return 0; }

    //! Return the process IDs and corresponding local IDs for the given global IDs.
    LookupStatus getRemoteIndexList(const Teuchos::ArrayView< const GlobalOrdinal > &GIDList, const Teuchos::ArrayView< int > &nodeIDList, const Teuchos::ArrayView< LocalOrdinal > &LIDList) const { return Xpetra::IDNotPresent; }

    //! Return the process IDs for the given global IDs.
    LookupStatus getRemoteIndexList(const Teuchos::ArrayView< const GlobalOrdinal > &GIDList, const Teuchos::ArrayView< int > &nodeIDList) const { return Xpetra::IDNotPresent; }

    //! Return a view of the global indices owned by this node.
    Teuchos::ArrayView< const GlobalOrdinal > getNodeElementList() const { return Teuchos::ArrayView<const GlobalOrdinal>(); }

    //@}

    //! @name Boolean tests
    //@{

    //! True if the local index is valid for this Map on this node, else false.
    bool isNodeLocalElement(LocalOrdinal localIndex) const { return false; }

    //! True if the global index is found in this Map on this node, else false.
    bool isNodeGlobalElement(GlobalOrdinal globalIndex) const { return false; }

    //! True if this Map is distributed contiguously, else false.
    bool isContiguous() const { return false; }

    //! Whether this Map is globally distributed or locally replicated.
    bool isDistributed() const { return false; }

    //! True if and only if map is compatible with this Map.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    bool isCompatible(const Map< LocalOrdinal, GlobalOrdinal, Node > &map) const { return false; }
#else
    bool isCompatible(const Map<Node > &map) const { return false; }
#endif

    //! True if and only if map is identical to this Map.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    bool isSameAs(const Map< LocalOrdinal, GlobalOrdinal, Node > &map) const { return false; }
#else
    bool isSameAs(const Map<Node > &map) const { return false; }
#endif

    //@}

    //! @name
    //@{

    //! Get this Map's Comm object.
    Teuchos::RCP< const Teuchos::Comm< int > >  getComm() const { return Teuchos::null; }


    //@}

    //! @name
    //@{

    //! Return a simple one-line description of this object.
    std::string description() const { return std::string(""); }

    //! Print this object with the given verbosity level to the given FancyOStream.
    void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel=Teuchos::Describable::verbLevel_default) const { }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > removeEmptyProcesses () const { return Teuchos::null; }
    RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > replaceCommWithSubset (const Teuchos::RCP<const Teuchos::Comm<int> >& newComm) const { return Teuchos::null; }
#else
    RCP<const Map<Node> > removeEmptyProcesses () const { return Teuchos::null; }
    RCP<const Map<Node> > replaceCommWithSubset (const Teuchos::RCP<const Teuchos::Comm<int> >& newComm) const { return Teuchos::null; }
#endif

//@}

    //! @name Xpetra specific
    //@{

    //! TpetraMap constructor to wrap a Tpetra::Map object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraMap(const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node > > &map) {
#else
    TpetraMap(const Teuchos::RCP<const Tpetra::Map<Node > > &map) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraMap<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraMap<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

    //! Get the library used by this object (Tpetra or Epetra?)
    UnderlyingLib lib() const { return UseTpetra; }

    //! Get the underlying Tpetra map
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Tpetra::Map< LocalOrdinal, GlobalOrdinal, Node > > getTpetra_Map() const { return Teuchos::null; }
#else
    RCP< const Tpetra::Map<Node > > getTpetra_Map() const { return Teuchos::null; }
#endif

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#ifdef HAVE_XPETRA_TPETRA
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    using local_map_type = typename Map<LocalOrdinal, GlobalOrdinal, Node>::local_map_type;
#else
    using local_map_type = typename Map<Node>::local_map_type;
#endif
    /// \brief Get the local Map for Kokkos kernels.
    local_map_type getLocalMap () const {
      // We will never be here, this is a stub class
      return local_map_type();
    }
#endif
#endif

    //@}
  }; // TpetraMap class (specialization for GO=int and NO=EpetraNode)
#endif

#endif // HAVE_XPETRA_EPETRA

} // Xpetra namespace

// TODO: remove?
//!  Returns true if \c map is identical to this map. Implemented in TpetraMap::isSameAs().
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class LocalOrdinal, class GlobalOrdinal, class Node>
bool operator== (const Xpetra::TpetraMap<LocalOrdinal,GlobalOrdinal,Node> &map1, const Xpetra::TpetraMap<LocalOrdinal,GlobalOrdinal,Node> &map2) {
#else
template <class Node>
bool operator== (const Xpetra::TpetraMap<Node> &map1, const Xpetra::TpetraMap<Node> &map2) {
#endif
  XPETRA_MONITOR("TpetraMap==TpetraMap");
  return map1.isSameAs(map2);
}

//! Returns true if \c map is not identical to this map. Implemented in TpetraMap::isSameAs().
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template <class LocalOrdinal, class GlobalOrdinal, class Node>
bool operator!= (const Xpetra::TpetraMap<LocalOrdinal,GlobalOrdinal,Node> &map1, const Xpetra::TpetraMap<LocalOrdinal,GlobalOrdinal,Node> &map2) {
#else
template <class Node>
bool operator!= (const Xpetra::TpetraMap<Node> &map1, const Xpetra::TpetraMap<Node> &map2) {
#endif
  XPETRA_MONITOR("TpetraMap!=TpetraMap");
  return !map1.isSameAs(map2);
}

#endif // XPETRA_TPETRAMAP_DEF_HPP

