#ifndef KOKKOS_FUSEDFUNCTOR
#define KOKKOS_FUSEDFUNCTOR

#include "Kokkos_Core.hpp"

/* A generalized functor that calls some other functors
 * in sequence, all using the same policy.
 *
 * This can save time on Kokkos/CUDA kernel launch overhead while
 * adding very little code (can call the FusedFunctor like a regular functor)
 */

namespace Kokkos
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

#endif

