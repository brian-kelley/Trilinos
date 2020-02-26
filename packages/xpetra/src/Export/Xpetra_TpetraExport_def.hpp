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
#ifndef XPETRA_TPETRAEXPORT_DEF_HPP
#define XPETRA_TPETRAEXPORT_DEF_HPP


#include "Xpetra_TpetraExport_decl.hpp"
#include "Tpetra_Distributor.hpp"

namespace Xpetra {

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Node>
TpetraExport<Node>::
#endif
TpetraExport(const Teuchos::RCP<const map_type>& source,
                                const Teuchos::RCP<const map_type>& target)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    : export_(Teuchos::rcp(new Tpetra::Export<LocalOrdinal, GlobalOrdinal, Node>(toTpetra(source), toTpetra(target))))
#else
    : export_(Teuchos::rcp(new Tpetra::Export<Node>(toTpetra(source), toTpetra(target))))
#endif
{
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Node>
TpetraExport<Node>::
#endif
TpetraExport(const Teuchos::RCP<const map_type>&                            source,
                                const Teuchos::RCP<const map_type>&         target,
                                const Teuchos::RCP<Teuchos::ParameterList>& plist)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    : export_(Teuchos::rcp(new Tpetra::Export<LocalOrdinal, GlobalOrdinal, Node>(toTpetra(source), toTpetra(target), plist)))
#else
    : export_(Teuchos::rcp(new Tpetra::Export<Node>(toTpetra(source), toTpetra(target), plist)))
#endif
{
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
TpetraExport(const Export<LocalOrdinal, GlobalOrdinal, Node>& rhs)
    : export_(Teuchos::rcp(new Tpetra::Export<LocalOrdinal, GlobalOrdinal, Node>(toTpetra(rhs))))
#else
template<class Node>
TpetraExport<Node>::
TpetraExport(const Export<Node>& rhs)
    : export_(Teuchos::rcp(new Tpetra::Export<Node>(toTpetra(rhs))))
#endif
{
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Node>
TpetraExport<Node>::
#endif
~TpetraExport()
{
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
size_t
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
TpetraExport<Node>::
#endif
getNumSameIDs() const
{
    XPETRA_MONITOR("TpetraExport::getNumSameIDs");
    return export_->getNumSameIDs();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
size_t
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
TpetraExport<Node>::
#endif
getNumPermuteIDs() const
{
    XPETRA_MONITOR("TpetraExport::getNumPermuteIDs");
    return export_->getNumPermuteIDs();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
ArrayView<const LocalOrdinal>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
TpetraExport<Node>::
#endif
getPermuteFromLIDs() const
{
    XPETRA_MONITOR("TpetraExport::getPermuteFromLIDs");
    return export_->getPermuteFromLIDs();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
ArrayView<const LocalOrdinal>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
TpetraExport<Node>::
#endif
getPermuteToLIDs() const
{
    XPETRA_MONITOR("TpetraExport::getPermuteToLIDs");
    return export_->getPermuteToLIDs();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
size_t
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
TpetraExport<Node>::
#endif
getNumRemoteIDs() const
{
    XPETRA_MONITOR("TpetraExport::getNumRemoteIDs");
    return export_->getNumRemoteIDs();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
ArrayView<const LocalOrdinal>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
TpetraExport<Node>::
#endif
getRemoteLIDs() const
{
    XPETRA_MONITOR("TpetraExport::getRemoteLIDs");
    return export_->getRemoteLIDs();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
size_t
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
TpetraExport<Node>::
#endif
getNumExportIDs() const
{
    XPETRA_MONITOR("TpetraExport::getNumExportIDs");
    return export_->getNumExportIDs();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
ArrayView<const LocalOrdinal>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
TpetraExport<Node>::
#endif
getExportLIDs() const
{
    XPETRA_MONITOR("TpetraExport::getExportLIDs");
    return export_->getExportLIDs();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
ArrayView<const int>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
TpetraExport<Node>::
#endif
getExportPIDs() const
{
    XPETRA_MONITOR("TpetraExport::getExportPIDs");
    return export_->getExportPIDs();
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node>>
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Node>
Teuchos::RCP<const Map<Node>>
TpetraExport<Node>::
#endif
getSourceMap() const
{
    XPETRA_MONITOR("TpetraExport::getSourceMap");
    return toXpetra(export_->getSourceMap());
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node>>
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Node>
Teuchos::RCP<const Map<Node>>
TpetraExport<Node>::
#endif
getTargetMap() const
{
    XPETRA_MONITOR("TpetraExport::getTargetMap");
    return toXpetra(export_->getTargetMap());
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
TpetraExport<Node>::
#endif
setDistributorParameters(const Teuchos::RCP<Teuchos::ParameterList> params) const {
  XPETRA_MONITOR("TpetraExport::setDistributorParameters");
  export_->getDistributor().setParameterList(params);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
#else
template<class Node>
#endif
void
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
TpetraExport<Node>::
#endif
print(std::ostream& os) const
{
    XPETRA_MONITOR("TpetraExport::print");
    export_->print(os);
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Node>
TpetraExport<Node>::
#endif
TpetraExport(
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  const RCP<const Tpetra::Export<LocalOrdinal, GlobalOrdinal, Node>>& exp)
#else
  const RCP<const Tpetra::Export<Node>>& exp)
#endif
    : export_(exp)
{
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Tpetra::Export<LocalOrdinal, GlobalOrdinal, Node>>
TpetraExport<LocalOrdinal, GlobalOrdinal, Node>::
#else
template<class Node>
RCP<const Tpetra::Export<Node>>
TpetraExport<Node>::
#endif
getTpetra_Export() const
{
    return export_;
}



#ifdef HAVE_XPETRA_EPETRA

#if((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT))) \
    || (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT))))

// stub implementation for GO=int and NO=EpetraNode
template<>
class TpetraExport<int, int, EpetraNode>
    : public Export<int, int, EpetraNode>
{

  public:
    typedef int        LocalOrdinal;
    typedef int        GlobalOrdinal;
    typedef EpetraNode Node;

    //! The specialization of Map used by this class.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Map<LocalOrdinal, GlobalOrdinal, Node> map_type;
#else
    typedef Map<Node> map_type;
#endif

    //! @name Constructor/Destructor Methods
    //@{

    //! Construct a Export object from the source and target Map.
    TpetraExport(const Teuchos::RCP<const map_type>& source, const Teuchos::RCP<const map_type>& target)
    {
        XPETRA_TPETRA_ETI_EXCEPTION(typeid(TpetraExport<LocalOrdinal, GlobalOrdinal, EpetraNode>).name(),
                                    typeid(TpetraExport<LocalOrdinal, GlobalOrdinal, EpetraNode>).name(),
                                    "int",
                                    typeid(EpetraNode).name());
    }


    //! Constructor (with list of parameters).
    TpetraExport(const Teuchos::RCP<const map_type>&         source,
                 const Teuchos::RCP<const map_type>&         target,
                 const Teuchos::RCP<Teuchos::ParameterList>& plist)
    {
        XPETRA_TPETRA_ETI_EXCEPTION(typeid(TpetraExport<LocalOrdinal, GlobalOrdinal, EpetraNode>).name(),
                                    typeid(TpetraExport<LocalOrdinal, GlobalOrdinal, EpetraNode>).name(),
                                    "int",
                                    typeid(EpetraNode).name());
    }


    //! Copy constructor.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraExport(const Export<LocalOrdinal, GlobalOrdinal, Node>& rhs)
#else
    TpetraExport(const Export<Node>& rhs)
#endif
    {
        XPETRA_TPETRA_ETI_EXCEPTION(typeid(TpetraExport<LocalOrdinal, GlobalOrdinal, EpetraNode>).name(),
                                    typeid(TpetraExport<LocalOrdinal, GlobalOrdinal, EpetraNode>).name(),
                                    "int",
                                    typeid(EpetraNode).name());
    }


    //! Destructor.
    ~TpetraExport() {}


    //@}

    //! @name Export Attribute Methods
    //@{


    //! Number of initial identical IDs.
    size_t getNumSameIDs() const { return 0; }


    //! Number of IDs to permute but not to communicate.
    size_t getNumPermuteIDs() const { return 0; }


    //! List of local IDs in the source Map that are permuted.
    ArrayView<const LocalOrdinal> getPermuteFromLIDs() const { return Teuchos::ArrayView<const LocalOrdinal>(); }


    //! List of local IDs in the target Map that are permuted.
    ArrayView<const LocalOrdinal> getPermuteToLIDs() const { return Teuchos::ArrayView<const LocalOrdinal>(); }


    //! Number of entries not on the calling process.
    size_t getNumRemoteIDs() const { return 0; }


    //! List of entries in the target Map to receive from other processes.
    ArrayView<const LocalOrdinal> getRemoteLIDs() const { return Teuchos::ArrayView<const LocalOrdinal>(); }


    //! Number of entries that must be sent by the calling process to other processes.
    size_t getNumExportIDs() const { return 0; }


    //! List of entries in the source Map that will be sent to other processes.
    ArrayView<const LocalOrdinal> getExportLIDs() const { return Teuchos::ArrayView<const LocalOrdinal>(); }


    //! List of processes to which entries will be sent.
    ArrayView<const int> getExportPIDs() const { return Teuchos::ArrayView<const int>(); }


    //! The source Map used to construct this Export.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node>> getSourceMap() const { return Teuchos::null; }
#else
    Teuchos::RCP<const Map<Node>> getSourceMap() const { return Teuchos::null; }
#endif


    //! The target Map used to construct this Export.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node>> getTargetMap() const { return Teuchos::null; }
#else
    Teuchos::RCP<const Map<Node>> getTargetMap() const { return Teuchos::null; }
#endif

    //! Set parameters on the underlying object
    void setDistributorParameters(const Teuchos::RCP<Teuchos::ParameterList> params) const { };


    //@}

    //! @name I/O Methods
    //@{


    //! Print the Export's data to the given output stream.
    void print(std::ostream& os) const
    { /* noop */
    }


    //@}


    //! @name Xpetra specific
    //@{


    //! TpetraExport constructor to wrap a Tpetra::Export object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraExport(const RCP<const Tpetra::Export<LocalOrdinal, GlobalOrdinal, Node>>& exp)
#else
    TpetraExport(const RCP<const Tpetra::Export<Node>>& exp)
#endif
    {
        XPETRA_TPETRA_ETI_EXCEPTION(typeid(TpetraExport<LocalOrdinal, GlobalOrdinal, EpetraNode>).name(),
                                    typeid(TpetraExport<LocalOrdinal, GlobalOrdinal, EpetraNode>).name(),
                                    "int",
                                    typeid(EpetraNode).name());
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<const Tpetra::Export<LocalOrdinal, GlobalOrdinal, Node>> getTpetra_Export() const { return Teuchos::null; }
#else
    RCP<const Tpetra::Export<Node>> getTpetra_Export() const { return Teuchos::null; }
#endif

    //@}

};      // TpetraExport class (specialization for LO=GO=int)
#endif      // #if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_INT)))



#if((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_LONG_LONG))) \
    || (!defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_LONG_LONG))))

// stub implementation for GO=long long and NO=EpetraNode
template<>
class TpetraExport<int, long long, EpetraNode>
    : public Export<int, long long, EpetraNode>
{

  public:
    typedef int        LocalOrdinal;
    typedef long long  GlobalOrdinal;
    typedef EpetraNode Node;

    //! The specialization of Map used by this class.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef Map<LocalOrdinal, GlobalOrdinal, Node> map_type;
#else
    typedef Map<Node> map_type;
#endif

    //! @name Constructor/Destructor Methods
    //@{


    //! Construct a Export object from the source and target Map.
    TpetraExport(const Teuchos::RCP<const map_type>& source, const Teuchos::RCP<const map_type>& target)
    {
        XPETRA_TPETRA_ETI_EXCEPTION(typeid(TpetraExport<LocalOrdinal, GlobalOrdinal, EpetraNode>).name(),
                                    typeid(TpetraExport<LocalOrdinal, GlobalOrdinal, EpetraNode>).name(),
                                    "long long",
                                    typeid(EpetraNode).name());
    }


    //! Constructor (with list of parameters).
    TpetraExport(const Teuchos::RCP<const map_type>&         source,
                 const Teuchos::RCP<const map_type>&         target,
                 const Teuchos::RCP<Teuchos::ParameterList>& plist)
    {
        XPETRA_TPETRA_ETI_EXCEPTION(typeid(TpetraExport<LocalOrdinal, GlobalOrdinal, EpetraNode>).name(),
                                    typeid(TpetraExport<LocalOrdinal, GlobalOrdinal, EpetraNode>).name(),
                                    "long long",
                                    typeid(EpetraNode).name());
    }


    //! Copy constructor.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraExport(const Export<LocalOrdinal, GlobalOrdinal, Node>& rhs)
#else
    TpetraExport(const Export<Node>& rhs)
#endif
    {
        XPETRA_TPETRA_ETI_EXCEPTION(typeid(TpetraExport<LocalOrdinal, GlobalOrdinal, EpetraNode>).name(),
                                    typeid(TpetraExport<LocalOrdinal, GlobalOrdinal, EpetraNode>).name(),
                                    "long long",
                                    typeid(EpetraNode).name());
    }


    //! Destructor.
    ~TpetraExport() {}


    //@}

    //! @name Export Attribute Methods
    //@{


    //! Number of initial identical IDs.
    size_t getNumSameIDs() const { return 0; }


    //! Number of IDs to permute but not to communicate.
    size_t getNumPermuteIDs() const { return 0; }


    //! List of local IDs in the source Map that are permuted.
    ArrayView<const LocalOrdinal> getPermuteFromLIDs() const { return Teuchos::ArrayView<const LocalOrdinal>(); }


    //! List of local IDs in the target Map that are permuted.
    ArrayView<const LocalOrdinal> getPermuteToLIDs() const { return Teuchos::ArrayView<const LocalOrdinal>(); }


    //! Number of entries not on the calling process.
    size_t getNumRemoteIDs() const { return 0; }


    //! List of entries in the target Map to receive from other processes.
    ArrayView<const LocalOrdinal> getRemoteLIDs() const { return Teuchos::ArrayView<const LocalOrdinal>(); }


    //! Number of entries that must be sent by the calling process to other processes.
    size_t getNumExportIDs() const { return 0; }


    //! List of entries in the source Map that will be sent to other processes.
    ArrayView<const LocalOrdinal> getExportLIDs() const { return Teuchos::ArrayView<const LocalOrdinal>(); }


    //! List of processes to which entries will be sent.
    ArrayView<const int> getExportPIDs() const { return Teuchos::ArrayView<const int>(); }


    //! The source Map used to construct this Export.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node>> getSourceMap() const { return Teuchos::null; }
#else
    Teuchos::RCP<const Map<Node>> getSourceMap() const { return Teuchos::null; }
#endif


    //! The target Map used to construct this Export.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node>> getTargetMap() const { return Teuchos::null; }
#else
    Teuchos::RCP<const Map<Node>> getTargetMap() const { return Teuchos::null; }
#endif

    //! Set parameters on the underlying object
    void setDistributorParameters(const Teuchos::RCP<Teuchos::ParameterList> params) const { };

    //@}

    //! @name I/O Methods
    //@{


    //! Print the Export's data to the given output stream.
    void print(std::ostream& os) const
    { /* noop */
    }


    //@}

    //! @name Xpetra specific
    //@{


    //! TpetraExport constructor to wrap a Tpetra::Export object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraExport(const RCP<const Tpetra::Export<LocalOrdinal, GlobalOrdinal, Node>>& exp) 
#else
    TpetraExport(const RCP<const Tpetra::Export<Node>>& exp) 
#endif
    {
        XPETRA_TPETRA_ETI_EXCEPTION(typeid(TpetraExport<LocalOrdinal, GlobalOrdinal, EpetraNode>).name(),
                                    typeid(TpetraExport<LocalOrdinal, GlobalOrdinal, EpetraNode>).name(),
                                    "long long",
                                    typeid(EpetraNode).name());
    }


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<const Tpetra::Export<LocalOrdinal, GlobalOrdinal, Node>> getTpetra_Export() const { return Teuchos::null; }
#else
    RCP<const Tpetra::Export<Node>> getTpetra_Export() const { return Teuchos::null; }
#endif


    //@}

};      // TpetraExport class (specialization for GO=long long, NO=EpetraNode)
#endif      // #if ((defined(EPETRA_HAVE_OMP) && (!defined(HAVE_TPETRA_INST_OPENMP) || !defined(HAVE_TPETRA_INST_INT_LONG_LONG)))


#endif      // HAVE_XPETRA_EPETRA


}      // namespace Xpetra


#endif      // XPETRA_TPETRAEXPORT_DEF_HPP


