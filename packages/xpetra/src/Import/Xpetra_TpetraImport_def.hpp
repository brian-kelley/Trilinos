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
#ifndef XPETRA_TPETRAIMPORT_DEF_HPP
#define XPETRA_TPETRAIMPORT_DEF_HPP
#include "Xpetra_TpetraConfigDefs.hpp"

#include "Xpetra_Import.hpp"
#include "Xpetra_TpetraImport_decl.hpp"
#include "Xpetra_Exceptions.hpp"

#include "Xpetra_TpetraMap.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Distributor.hpp"

namespace Xpetra {

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::TpetraImport(const Teuchos::RCP< const map_type > &source, const Teuchos::RCP< const map_type > &target):import_(Teuchos::rcp(new Tpetra::Import< LocalOrdinal, GlobalOrdinal, Node >(toTpetra(source), toTpetra(target))))
#else
template<class Node>
TpetraImport<Node>::TpetraImport(const Teuchos::RCP< const map_type > &source, const Teuchos::RCP< const map_type > &target):import_(Teuchos::rcp(new Tpetra::Import<Node >(toTpetra(source), toTpetra(target))))
#endif
{   }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::TpetraImport(const Teuchos::RCP< const map_type > &source, const Teuchos::RCP< const map_type > &target, const Teuchos::RCP< Teuchos::ParameterList > &plist):import_(Teuchos::rcp(new Tpetra::Import< LocalOrdinal, GlobalOrdinal, Node >(toTpetra(source), toTpetra(target), plist)))
#else
template<class Node>
TpetraImport<Node>::TpetraImport(const Teuchos::RCP< const map_type > &source, const Teuchos::RCP< const map_type > &target, const Teuchos::RCP< Teuchos::ParameterList > &plist):import_(Teuchos::rcp(new Tpetra::Import<Node >(toTpetra(source), toTpetra(target), plist)))
#endif
{   }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::TpetraImport(const Import< LocalOrdinal, GlobalOrdinal, Node > &import):import_(Teuchos::rcp(new Tpetra::Import< LocalOrdinal, GlobalOrdinal, Node >(toTpetra(import))))
#else
template<class Node>
TpetraImport<Node>::TpetraImport(const Import<Node > &import):import_(Teuchos::rcp(new Tpetra::Import<Node >(toTpetra(import))))
#endif
{   }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::~TpetraImport()
#else
template<class Node>
TpetraImport<Node>::~TpetraImport()
#endif
{  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
size_t TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::getNumSameIDs() const
#else
template<class Node>
size_t TpetraImport<Node>::getNumSameIDs() const
#endif
{ XPETRA_MONITOR("TpetraImport::getNumSameIDs"); return import_->getNumSameIDs(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
size_t TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::getNumPermuteIDs() const
#else
template<class Node>
size_t TpetraImport<Node>::getNumPermuteIDs() const
#endif
{ XPETRA_MONITOR("TpetraImport::getNumPermuteIDs"); return import_->getNumPermuteIDs(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
ArrayView< const LocalOrdinal > TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::getPermuteFromLIDs() const
#else
template<class Node>
ArrayView< const LocalOrdinal > TpetraImport<Node>::getPermuteFromLIDs() const
#endif
{ XPETRA_MONITOR("TpetraImport::getPermuteFromLIDs"); return import_->getPermuteFromLIDs(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
ArrayView< const LocalOrdinal > TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::getPermuteToLIDs() const
#else
template<class Node>
ArrayView< const LocalOrdinal > TpetraImport<Node>::getPermuteToLIDs() const
#endif
{ XPETRA_MONITOR("TpetraImport::getPermuteToLIDs"); return import_->getPermuteToLIDs(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
size_t TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::getNumRemoteIDs() const
#else
template<class Node>
size_t TpetraImport<Node>::getNumRemoteIDs() const
#endif
{ XPETRA_MONITOR("TpetraImport::getNumRemoteIDs"); return import_->getNumRemoteIDs(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::setDistributorParameters(const Teuchos::RCP<Teuchos::ParameterList> params) const
#else
template<class Node>
void TpetraImport<Node>::setDistributorParameters(const Teuchos::RCP<Teuchos::ParameterList> params) const
#endif
{ XPETRA_MONITOR("TpetraImport::setDistributorParameters"); import_->getDistributor().setParameterList(params); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
ArrayView< const LocalOrdinal > TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::getRemoteLIDs() const
#else
template<class Node>
ArrayView< const LocalOrdinal > TpetraImport<Node>::getRemoteLIDs() const
#endif
{ XPETRA_MONITOR("TpetraImport::getRemoteLIDs"); return import_->getRemoteLIDs(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
size_t TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::getNumExportIDs() const
#else
template<class Node>
size_t TpetraImport<Node>::getNumExportIDs() const
#endif
{ XPETRA_MONITOR("TpetraImport::getNumExportIDs"); return import_->getNumExportIDs(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
ArrayView< const LocalOrdinal > TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::getExportLIDs() const
#else
template<class Node>
ArrayView< const LocalOrdinal > TpetraImport<Node>::getExportLIDs() const
#endif
{ XPETRA_MONITOR("TpetraImport::getExportLIDs"); return import_->getExportLIDs(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
ArrayView< const int > TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::getExportPIDs() const
#else
template<class Node>
ArrayView< const int > TpetraImport<Node>::getExportPIDs() const
#endif
{ XPETRA_MONITOR("TpetraImport::getExportPIDs"); return import_->getExportPIDs(); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::getSourceMap() const
#else
template<class Node>
Teuchos::RCP< const Map<Node > > TpetraImport<Node>::getSourceMap() const
#endif
{ XPETRA_MONITOR("TpetraImport::getSourceMap"); return toXpetra(import_->getSourceMap()); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::getTargetMap() const
#else
template<class Node>
Teuchos::RCP< const Map<Node > > TpetraImport<Node>::getTargetMap() const
#endif
{ XPETRA_MONITOR("TpetraImport::getTargetMap"); return toXpetra(import_->getTargetMap()); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::print(std::ostream &os) const
#else
template<class Node>
void TpetraImport<Node>::print(std::ostream &os) const
#endif
{ XPETRA_MONITOR("TpetraImport::print"); import_->print(os); }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::TpetraImport(const RCP<const Tpetra::Import< LocalOrdinal, GlobalOrdinal, Node > > &import) : import_(import)
#else
template<class Node>
TpetraImport<Node>::TpetraImport(const RCP<const Tpetra::Import<Node > > &import) : import_(import)
#endif
{  }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
RCP< const Tpetra::Import< LocalOrdinal, GlobalOrdinal, Node > > TpetraImport<LocalOrdinal,GlobalOrdinal,Node>::getTpetra_Import() const
#else
template<class Node>
RCP< const Tpetra::Import<Node > > TpetraImport<Node>::getTpetra_Import() const
#endif
{ return import_; }


#ifdef HAVE_XPETRA_EPETRA

#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))

  // stub implementation for GO=int and NO=EpetraNode
  template <>
  class TpetraImport<int, int, EpetraNode> : public Import<int, int, EpetraNode>
  {

  public:
    typedef int LocalOrdinal;
    typedef int GlobalOrdinal;
    typedef EpetraNode Node;

    //! The specialization of Map used by this class.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Map<LocalOrdinal,GlobalOrdinal,Node> map_type;
#else
    typedef Map<Node> map_type;
#endif

    //! @name Constructor/Destructor Methods
    //@{

    //! Construct an Import from the source and target Maps.
    TpetraImport(const Teuchos::RCP< const map_type > &source, const Teuchos::RCP< const map_type > &target) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraImport<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraImport<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Constructor (with list of parameters).
    TpetraImport(const Teuchos::RCP< const map_type > &source, const Teuchos::RCP< const map_type > &target, const Teuchos::RCP< Teuchos::ParameterList > &plist) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraImport<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraImport<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Copy constructor.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraImport(const Import< LocalOrdinal, GlobalOrdinal, Node > &import) {
#else
    TpetraImport(const Import<Node > &import) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraImport<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraImport<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

    //! Destructor.
    ~TpetraImport() {  }

    //@}

    //! @name Import Attribute Methods
    //@{

    //! Number of initial identical IDs.
    size_t getNumSameIDs() const { return 0; }

    //! Number of IDs to permute but not to communicate.
    size_t getNumPermuteIDs() const { return 0; }

    //! List of local IDs in the source Map that are permuted.
    ArrayView< const LocalOrdinal > getPermuteFromLIDs() const { return Teuchos::ArrayView<const LocalOrdinal>(); }

    //! List of local IDs in the target Map that are permuted.
    ArrayView< const LocalOrdinal > getPermuteToLIDs() const { return Teuchos::ArrayView<const LocalOrdinal>();  }

    //! Number of entries not on the calling process.
    size_t getNumRemoteIDs() const { return 0; }

    //! List of entries in the target Map to receive from other processes.
    ArrayView< const LocalOrdinal > getRemoteLIDs() const { return Teuchos::ArrayView<const LocalOrdinal>();  }

    //! Number of entries that must be sent by the calling process to other processes.
    size_t getNumExportIDs() const { return 0; }

    //! List of entries in the source Map that will be sent to other processes.
    ArrayView< const LocalOrdinal > getExportLIDs() const { return Teuchos::ArrayView<const LocalOrdinal>();  }

    //! List of processes to which entries will be sent.
    ArrayView< const int > getExportPIDs() const { return Teuchos::ArrayView<const int>();  }

    //! The Source Map used to construct this Import object.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > getSourceMap() const { return Teuchos::null; }
#else
    Teuchos::RCP< const Map<Node > > getSourceMap() const { return Teuchos::null; }
#endif

    //! The Target Map used to construct this Import object.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > getTargetMap() const { return Teuchos::null; }
#else
    Teuchos::RCP< const Map<Node > > getTargetMap() const { return Teuchos::null; }
#endif

    //! Set parameters on the underlying object
    void setDistributorParameters(const Teuchos::RCP<Teuchos::ParameterList> params) const { }

    //@}

    //! @name I/O Methods
    //@{

    //! Print the Import's data to the given output stream.
    void print(std::ostream &os) const { /* noop */ }

    //@}

    //! @name Xpetra specific
    //@{

    //! TpetraImport constructor to wrap a Tpetra::Import object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraImport(const RCP<const Tpetra::Import< LocalOrdinal, GlobalOrdinal, Node > > &import) {
#else
    TpetraImport(const RCP<const Tpetra::Import<Node > > &import) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraImport<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraImport<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "int", typeid(EpetraNode).name() );
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Tpetra::Import< LocalOrdinal, GlobalOrdinal, Node > > getTpetra_Import() const { return Teuchos::null; }
#else
    RCP< const Tpetra::Import<Node > > getTpetra_Import() const { return Teuchos::null; }
#endif

    //@}

  }; // TpetraImport class (stub implementation for GO=int, NO=EpetraNode)
#endif

#if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_LONG_LONG))) || \
    (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_LONG_LONG))))

  // stub implementation for GO=long long and NO=EpetraNode
  template <>
  class TpetraImport<int, long long, EpetraNode> : public Import<int, long long, EpetraNode>
  {

  public:
    typedef int LocalOrdinal;
    typedef long long GlobalOrdinal;
    typedef EpetraNode Node;

    //! The specialization of Map used by this class.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Map<LocalOrdinal,GlobalOrdinal,Node> map_type;
#else
    typedef Map<Node> map_type;
#endif

    //! @name Constructor/Destructor Methods
    //@{

    //! Construct an Import from the source and target Maps.
    TpetraImport(const Teuchos::RCP< const map_type > &source, const Teuchos::RCP< const map_type > &target) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraImport<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraImport<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

    //! Constructor (with list of parameters).
    TpetraImport(const Teuchos::RCP< const map_type > &source, const Teuchos::RCP< const map_type > &target, const Teuchos::RCP< Teuchos::ParameterList > &plist) {
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraImport<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraImport<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

    //! Copy constructor.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraImport(const Import< LocalOrdinal, GlobalOrdinal, Node > &import) {
#else
    TpetraImport(const Import<Node > &import) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraImport<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraImport<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

    //! Destructor.
    ~TpetraImport() {  }

    //@}

    //! @name Import Attribute Methods
    //@{

    //! Number of initial identical IDs.
    size_t getNumSameIDs() const { return 0; }

    //! Number of IDs to permute but not to communicate.
    size_t getNumPermuteIDs() const { return 0; }

    //! List of local IDs in the source Map that are permuted.
    ArrayView< const LocalOrdinal > getPermuteFromLIDs() const { return Teuchos::ArrayView<const LocalOrdinal>(); }

    //! List of local IDs in the target Map that are permuted.
    ArrayView< const LocalOrdinal > getPermuteToLIDs() const { return Teuchos::ArrayView<const LocalOrdinal>();  }

    //! Number of entries not on the calling process.
    size_t getNumRemoteIDs() const { return 0; }

    //! List of entries in the target Map to receive from other processes.
    ArrayView< const LocalOrdinal > getRemoteLIDs() const { return Teuchos::ArrayView<const LocalOrdinal>();  }

    //! Number of entries that must be sent by the calling process to other processes.
    size_t getNumExportIDs() const { return 0; }

    //! List of entries in the source Map that will be sent to other processes.
    ArrayView< const LocalOrdinal > getExportLIDs() const { return Teuchos::ArrayView<const LocalOrdinal>();  }

    //! List of processes to which entries will be sent.
    ArrayView< const int > getExportPIDs() const { return Teuchos::ArrayView<const int>();  }

    //! The Source Map used to construct this Import object.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > getSourceMap() const { return Teuchos::null; }
#else
    Teuchos::RCP< const Map<Node > > getSourceMap() const { return Teuchos::null; }
#endif

    //! The Target Map used to construct this Import object.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > getTargetMap() const { return Teuchos::null; }
#else
    Teuchos::RCP< const Map<Node > > getTargetMap() const { return Teuchos::null; }
#endif

    //! Set parameters on the underlying object
    void setDistributorParameters(const Teuchos::RCP<Teuchos::ParameterList> params) const { }

    //@}

    //! @name I/O Methods
    //@{

    //! Print the Import's data to the given output stream.
    void print(std::ostream &os) const { /* noop */ }

    //@}

    //! @name Xpetra specific
    //@{

    //! TpetraImport constructor to wrap a Tpetra::Import object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraImport(const RCP<const Tpetra::Import< LocalOrdinal, GlobalOrdinal, Node > > &import) {
#else
    TpetraImport(const RCP<const Tpetra::Import<Node > > &import) {
#endif
      XPETRA_TPETRA_ETI_EXCEPTION( typeid(TpetraImport<LocalOrdinal,GlobalOrdinal,EpetraNode>).name() , typeid(TpetraImport<LocalOrdinal,GlobalOrdinal,EpetraNode>).name(), "long long", typeid(EpetraNode).name() );
    }

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP< const Tpetra::Import< LocalOrdinal, GlobalOrdinal, Node > > getTpetra_Import() const { return Teuchos::null; }
#else
    RCP< const Tpetra::Import<Node > > getTpetra_Import() const { return Teuchos::null; }
#endif

    //@}

  }; // TpetraImport class (stub implementation for GO=long long, NO=EpetraNode)
#endif

#endif // HAVE_XPETRA_EPETRA

} // Xpetra namespace

#endif
