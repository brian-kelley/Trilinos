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
#ifndef XPETRA_TPETRAVECTOR_DECL_HPP
#define XPETRA_TPETRAVECTOR_DECL_HPP

#include "Xpetra_TpetraConfigDefs.hpp"

#include "Xpetra_Vector.hpp"
#include "Xpetra_MultiVector_decl.hpp"
#include "Xpetra_TpetraMultiVector_decl.hpp"

#include "Xpetra_TpetraMap_decl.hpp" 
#include "Xpetra_Utils.hpp"

#include "Tpetra_Vector.hpp"

namespace Xpetra {

// TODO: move that elsewhere
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> toTpetra(Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>&);
#else
template<class Scalar, class Node>
RCP<Tpetra::Vector<Scalar, Node>> toTpetra(Vector<Scalar, Node>&);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> toTpetra(const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>&);
#else
template<class Scalar, class Node>
RCP<Tpetra::Vector<Scalar, Node>> toTpetra(const Vector<Scalar, Node>&);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> toXpetra(RCP<const Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> vec);
#else
template<class Scalar, class Node>
RCP<const Vector<Scalar, Node>> toXpetra(RCP<const Tpetra::Vector<Scalar, Node>> vec);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> toXpetra(RCP<Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> vec);
#else
template<class Scalar, class Node>
RCP<Vector<Scalar, Node>> toXpetra(RCP<Tpetra::Vector<Scalar, Node>> vec);
#endif

//
//

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node = KokkosClassic::DefaultNode::DefaultNodeType>
#else
template<class Scalar, class Node = KokkosClassic::DefaultNode::DefaultNodeType>
#endif
class TpetraVector
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    : public virtual Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>
    , public TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>
#else
    : public virtual Vector<Scalar, Node>
    , public TpetraMultiVector<Scalar, Node>
#endif
{

  public:
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    using TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::dot;                     // overloading, not hiding
    using TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::norm1;                   // overloading, not hiding
    using TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::norm2;                   // overloading, not hiding
    using TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::normInf;                 // overloading, not hiding
    using TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::meanValue;               // overloading, not hiding
    using TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::replaceGlobalValue;      // overloading, not hiding
    using TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::sumIntoGlobalValue;      // overloading, not hiding
    using TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::replaceLocalValue;       // overloading, not hiding
    using TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::sumIntoLocalValue;       // overloading, not hiding
#else
    using LocalOrdinal = typename Tpetra::Map<>::local_ordinal_type;
    using GlobalOrdinal = typename Tpetra::Map<>::global_ordinal_type;
    using TpetraMultiVector<Scalar, Node>::dot;                     // overloading, not hiding
    using TpetraMultiVector<Scalar, Node>::norm1;                   // overloading, not hiding
    using TpetraMultiVector<Scalar, Node>::norm2;                   // overloading, not hiding
    using TpetraMultiVector<Scalar, Node>::normInf;                 // overloading, not hiding
    using TpetraMultiVector<Scalar, Node>::meanValue;               // overloading, not hiding
    using TpetraMultiVector<Scalar, Node>::replaceGlobalValue;      // overloading, not hiding
    using TpetraMultiVector<Scalar, Node>::sumIntoGlobalValue;      // overloading, not hiding
    using TpetraMultiVector<Scalar, Node>::replaceLocalValue;       // overloading, not hiding
    using TpetraMultiVector<Scalar, Node>::sumIntoLocalValue;       // overloading, not hiding
#endif

    //! @name Constructor/Destructor Methods
    //@{

    //! Sets all vector entries to zero.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraVector(const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node>>& map, bool zeroOut = true);
#else
    TpetraVector(const Teuchos::RCP<const Map<Node>>& map, bool zeroOut = true);
#endif

    //! Set multi-vector values from an array using Teuchos memory management classes. (copy)
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraVector(const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node>>& map, const Teuchos::ArrayView<const Scalar>& A);
#else
    TpetraVector(const Teuchos::RCP<const Map<Node>>& map, const Teuchos::ArrayView<const Scalar>& A);
#endif

    //! Destructor.
    virtual ~TpetraVector();

    //@}

    //! @name Post-construction modification routines
    //@{

    //! Replace current value at the specified location with specified value.
    void replaceGlobalValue(GlobalOrdinal globalRow, const Scalar& value);

    //! Adds specified value to existing value at the specified location.
    void sumIntoGlobalValue(GlobalOrdinal globalRow, const Scalar& value);

    //! Replace current value at the specified location with specified values.
    void replaceLocalValue(LocalOrdinal myRow, const Scalar& value);

    //! Adds specified value to existing value at the specified location.
    void sumIntoLocalValue(LocalOrdinal myRow, const Scalar& value);

    //@}

    //! @name Mathematical methods
    //@{

    //! Return 1-norm of this Vector.
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType norm1() const;

    //! Compute 2-norm of this Vector.
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType norm2() const;

    //! Compute Inf-norm of this Vector.
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType normInf() const;

    //! Compute mean (average) value of this Vector.
    Scalar meanValue() const;

    //@}

    //! @name Overridden from Teuchos::Describable
    //@{

    //! Return a simple one-line description of this object.
    std::string description() const;

    //! Print the object with some verbosity level to an FancyOStream object.
    void describe(Teuchos::FancyOStream& out, const Teuchos::EVerbosityLevel verbLevel = Teuchos::Describable::verbLevel_default) const;

    //@}

    //! Computes dot product of this Vector against input Vector x.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    Scalar dot(const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& a) const;
#else
    Scalar dot(const Vector<Scalar, Node>& a) const;
#endif

//! @name Xpetra specific
    //@{

    //! TpetraMultiVector constructor to wrap a Tpetra::MultiVector object
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    TpetraVector(const Teuchos::RCP<Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>& vec);
#else
    TpetraVector(const Teuchos::RCP<Tpetra::Vector<Scalar, Node>>& vec);
#endif

    //! Get the underlying Tpetra multivector
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    RCP<Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> getTpetra_Vector() const;
#else
    RCP<Tpetra::Vector<Scalar, Node>> getTpetra_Vector() const;
#endif

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef typename Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::dual_view_type dual_view_type;
#else
    typedef typename Xpetra::MultiVector<Scalar, Node>::dual_view_type dual_view_type;
#endif

    typename dual_view_type::t_host_um getHostLocalView() const;

    typename dual_view_type::t_dev_um getDeviceLocalView() const;

    /// \brief Return an unmanaged non-const view of the local data on a specific device.
    /// \tparam TargetDeviceType The Kokkos Device type whose data to return.
    ///
    /// \warning DO NOT USE THIS FUNCTION! There is no reason why you are working directly
    ///          with the Xpetra::TpetraVector object. To write a code which is independent
    ///          from the underlying linear algebra package you should always use the abstract class,
    ///          i.e. Xpetra::Vector!
    ///
    /// \warning Be aware that the view on the vector data is non-persisting, i.e.
    ///          only valid as long as the vector does not run of scope!
    template<class TargetDeviceType>
    typename Kokkos::Impl::if_c<
      Kokkos::Impl::is_same<typename dual_view_type::t_dev_um::execution_space::memory_space, typename TargetDeviceType::memory_space>::value,
      typename dual_view_type::t_dev_um,
      typename dual_view_type::t_host_um>::type
    getLocalView() const;
#endif

    //@}

};      // TpetraVector class


// TODO: move that elsewhere
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
toTpetra(Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& x)
#else
template<class Scalar, class Node>
RCP<Tpetra::Vector<Scalar, Node>>
toTpetra(Vector<Scalar, Node>& x)
#endif
{
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef TpetraVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> TpetraVectorClass;
#else
    typedef TpetraVector<Scalar, Node> TpetraVectorClass;
#endif
    XPETRA_DYNAMIC_CAST(TpetraVectorClass, x, tX, "toTpetra");
    return tX.getTpetra_Vector();
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
toTpetra(const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& x)
#else
template<class Scalar, class Node>
RCP<Tpetra::Vector<Scalar, Node>>
toTpetra(const Vector<Scalar, Node>& x)
#endif
{
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    typedef TpetraVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> TpetraVectorClass;
#else
    typedef TpetraVector<Scalar, Node> TpetraVectorClass;
#endif
    XPETRA_DYNAMIC_CAST(const TpetraVectorClass, x, tX, "toTpetra");
    return tX.getTpetra_Vector();
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
toXpetra(RCP<Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> vec)
#else
template<class Scalar, class Node>
RCP<Vector<Scalar, Node>>
toXpetra(RCP<Tpetra::Vector<Scalar, Node>> vec)
#endif
{
    if(!vec.is_null())
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        return rcp(new TpetraVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>(vec));
#else
        return rcp(new TpetraVector<Scalar, Node>(vec));
#endif

    return Teuchos::null;
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
toXpetra(RCP<const Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> vec)
#else
template<class Scalar, class Node>
RCP<const Vector<Scalar, Node>>
toXpetra(RCP<const Tpetra::Vector<Scalar, Node>> vec)
#endif
{
    // We cast away the const to wrap the Tpetra vector into an Xpetra object. But it's OK because the Xpetra vector is returned as const.
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    return toXpetra(Teuchos::rcp_const_cast<Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>(vec));
#else
    return toXpetra(Teuchos::rcp_const_cast<Tpetra::Vector<Scalar, Node>>(vec));
#endif
}

}      // namespace Xpetra

#define XPETRA_TPETRAVECTOR_SHORT
#endif      // XPETRA_TPETRAVECTOR_DECL_HPP
