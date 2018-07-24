/*
//@HEADER
// ***********************************************************************
//
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//                 Copyright (2009) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// ***********************************************************************
//@HEADER
*/

#ifndef IFPACK2_FUSEDFUNCTOR 
#define IFPACK2_FUSEDFUNCTOR 

#include "Kokkos_Core.hpp"

/* A generalized functor that calls some other functors
 * in sequence, all using the same policy.
 *
 * This can save time on Kokkos/CUDA kernel launch overhead while
 * adding very little code (can call the FusedFunctor like a regular functor)
 */

namespace Ifpack2 
{
namespace Details
{
  template<typename...Args> struct FusedFunctor;

  //Specialization for 0 functors (no-op)
  template<typename Policy>
  struct FusedFunctor<Policy>
  {
    typedef typename Policy::member_type member_type;
    KOKKOS_INLINE_FUNCTION void operator()(const member_type) const {}
  };

  //Specialization that executes 1 functor and then 0 or more others
  template<typename Policy, typename HeadFunctor, typename...TailFunctors>
  struct FusedFunctor<Policy, HeadFunctor, TailFunctors...>
  : public FusedFunctor<Policy, TailFunctors...>
  {
    typedef FusedFunctor<Policy, TailFunctors...> Parent;
    FusedFunctor(HeadFunctor& head_, TailFunctors...tail) : Parent(tail...), head(head_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const typename Parent::member_type i) const
    {
      head(i);
      Parent::operator()(i);
    }
    HeadFunctor head;
  };

  template<typename Policy>
  struct fuseFunctors
  {
    template<typename...Args>
    FusedFunctor<Policy, Args...> operator()(Args...args)
    {
      return FusedFunctor<Policy, Args...>(args...);
    };
  };
}
}

#endif

