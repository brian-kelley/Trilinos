// @HEADER
// *****************************************************************************
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//
// Copyright 2009 NTESS and the Ifpack2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef IFPACK2_BLOCKCOMPUTERES_VECTOR_DEF_HPP
#define IFPACK2_BLOCKCOMPUTERES_VECTOR_DEF_HPP

#include "Ifpack2_BlockComputeResidualVector_decl.hpp"

namespace Ifpack2::BlockHelperDetails {

///
/// compute local residula vector y = b - R x
///
static inline int ComputeResidualVectorRecommendedCudaVectorSize(const int blksize,
                                                                 const int team_size) {
  int total_team_size(0);
  if (blksize <= 5)
    total_team_size = 32;
  else if (blksize <= 9)
    total_team_size = 32;  // 64
  else if (blksize <= 12)
    total_team_size = 96;
  else if (blksize <= 16)
    total_team_size = 128;
  else if (blksize <= 20)
    total_team_size = 160;
  else
    total_team_size = 160;
  return total_team_size / team_size;
}

static inline int ComputeResidualVectorRecommendedHIPVectorSize(const int blksize,
                                                                const int team_size) {
  int total_team_size(0);
  if (blksize <= 5)
    total_team_size = 32;
  else if (blksize <= 9)
    total_team_size = 32;  // 64
  else if (blksize <= 12)
    total_team_size = 96;
  else if (blksize <= 16)
    total_team_size = 128;
  else if (blksize <= 20)
    total_team_size = 160;
  else
    total_team_size = 160;
  return total_team_size / team_size;
}

static inline int ComputeResidualVectorRecommendedSYCLVectorSize(const int blksize,
                                                                 const int team_size) {
  int total_team_size(0);
  if (blksize <= 5)
    total_team_size = 32;
  else if (blksize <= 9)
    total_team_size = 32;  // 64
  else if (blksize <= 12)
    total_team_size = 96;
  else if (blksize <= 16)
    total_team_size = 128;
  else if (blksize <= 20)
    total_team_size = 160;
  else
    total_team_size = 160;
  return total_team_size / team_size;
}

template <typename T>
static inline int ComputeResidualVectorRecommendedVectorSize(const int blksize,
                                                             const int team_size) {
  if (is_cuda<T>::value)
    return ComputeResidualVectorRecommendedCudaVectorSize(blksize, team_size);
  if (is_hip<T>::value)
    return ComputeResidualVectorRecommendedHIPVectorSize(blksize, team_size);
  if (is_sycl<T>::value)
    return ComputeResidualVectorRecommendedSYCLVectorSize(blksize, team_size);
  return -1;
}

template <typename MatrixType>
struct ComputeResidualFunctor {
  using impl_type        = BlockHelperDetails::ImplType<MatrixType>;
  using node_device_type = typename impl_type::node_device_type;
  using execution_space  = typename impl_type::execution_space;
  using memory_space     = typename impl_type::memory_space;

  using local_ordinal_type  = typename impl_type::local_ordinal_type;
  using size_type           = typename impl_type::size_type;
  using impl_scalar_type    = typename impl_type::impl_scalar_type;
  using magnitude_type      = typename impl_type::magnitude_type;
  using btdm_scalar_type    = typename impl_type::btdm_scalar_type;
  using btdm_magnitude_type = typename impl_type::btdm_magnitude_type;
  /// views
  using local_ordinal_type_1d_view      = typename impl_type::local_ordinal_type_1d_view;
  using size_type_1d_view               = typename impl_type::size_type_1d_view;
  using tpetra_block_access_view_type   = typename impl_type::tpetra_block_access_view_type;  // block crs (layout right)
  using impl_scalar_type_1d_view        = typename impl_type::impl_scalar_type_1d_view;
  using impl_scalar_type_2d_view_tpetra = typename impl_type::impl_scalar_type_2d_view_tpetra;  // block multivector (layout left)
  using vector_type_3d_view             = typename impl_type::vector_type_3d_view;
  using btdm_scalar_type_4d_view        = typename impl_type::btdm_scalar_type_4d_view;
  using i64_3d_view                     = typename impl_type::i64_3d_view;
  static constexpr int vector_length    = impl_type::vector_length;

  /// team policy member type (used in cuda)
  using member_type = typename Kokkos::TeamPolicy<execution_space>::member_type;

  ConstUnmanaged<impl_scalar_type_2d_view_tpetra> b;
  ConstUnmanaged<impl_scalar_type_2d_view_tpetra> x;  // x_owned
  ConstUnmanaged<impl_scalar_type_2d_view_tpetra> x_remote;
  Unmanaged<impl_scalar_type_2d_view_tpetra> y;
  Unmanaged<vector_type_3d_view> y_packed;
  Unmanaged<btdm_scalar_type_4d_view> y_packed_scalar;

  // AmD information
  const ConstUnmanaged<size_type_1d_view> rowptr, rowptr_remote;
  const ConstUnmanaged<local_ordinal_type_1d_view> colindsub, colindsub_remote;
  const ConstUnmanaged<impl_scalar_type_1d_view> tpetra_values;

  // block crs graph information
  // for cuda (kokkos crs graph uses a different size_type from size_t)
  const ConstUnmanaged<Kokkos::View<size_t *, node_device_type>> A_block_rowptr;
  const ConstUnmanaged<Kokkos::View<size_t *, node_device_type>> A_point_rowptr;
  const ConstUnmanaged<Kokkos::View<local_ordinal_type *, node_device_type>> A_colind;

  // blocksize
  const local_ordinal_type blocksize_requested;

  // part interface
  const ConstUnmanaged<local_ordinal_type_1d_view> part2packrowidx0;
  const ConstUnmanaged<local_ordinal_type_1d_view> part2rowidx0;
  const ConstUnmanaged<local_ordinal_type_1d_view> rowidx2part;
  const ConstUnmanaged<local_ordinal_type_1d_view> partptr;
  const ConstUnmanaged<local_ordinal_type_1d_view> lclrow;
  const ConstUnmanaged<local_ordinal_type_1d_view> dm2cm;

  const bool is_dm2cm_active;
  const bool hasBlockCrsMatrix;

  ComputeResidualFunctor(const ComputeResidualVector<MatrixType> &crv)
    : rowptr(crv.rowptr)
    , rowptr_remote(crv.rowptr_remote)
    , colindsub(crv.colindsub)
    , colindsub_remote(crv.colindsub_remote)
    , tpetra_values(crv.tpetra_values)
    , A_block_rowptr(crv.A_block_rowptr)
    , A_point_rowptr(crv.A_point_rowptr)
    , A_colind(crv.A_colind)
    , blocksize_requested(crv.blocksize_requested)
    , part2packrowidx0(crv.part2packrowidx0)
    , part2rowidx0(crv.part2rowidx0)
    , rowidx2part(crv.rowidx2part)
    , partptr(crv.partptr)
    , lclrow(crv.lclrow)
    , dm2cm(crv.dm2cm)
    , is_dm2cm_active(crv.is_dm2cm_active)
    , hasBlockCrsMatrix(crv.hasBlockCrsMatrix) {}

  inline void
  SerialDot(const local_ordinal_type &blocksize,
            const local_ordinal_type &lclRowID,
            const local_ordinal_type &lclColID,
            const local_ordinal_type &ii,
            const ConstUnmanaged<local_ordinal_type_1d_view> colindsub_,
            const impl_scalar_type *const KOKKOS_RESTRICT xx,
            /* */ impl_scalar_type *KOKKOS_RESTRICT yy) const {
    const size_type Aj_c  = colindsub_(lclColID);
    auto point_row_offset = A_point_rowptr(lclRowID * blocksize + ii) + Aj_c * blocksize;
    impl_scalar_type val  = 0;
#if defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#pragma ivdep
#endif
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
    for (local_ordinal_type k1 = 0; k1 < blocksize; ++k1)
      val += tpetra_values(point_row_offset + k1) * xx[k1];
    yy[ii] -= val;
  }

  inline void
  SerialGemv(const local_ordinal_type &blocksize,
             const impl_scalar_type *const KOKKOS_RESTRICT AA,
             const impl_scalar_type *const KOKKOS_RESTRICT xx,
             /* */ impl_scalar_type *KOKKOS_RESTRICT yy) const {
    using tlb = BlockHelperDetails::TpetraLittleBlock<Tpetra::Impl::BlockCrsMatrixLittleBlockArrayLayout>;
    for (local_ordinal_type k0 = 0; k0 < blocksize; ++k0) {
      impl_scalar_type val = 0;
#if defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#pragma ivdep
#endif
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
      for (local_ordinal_type k1 = 0; k1 < blocksize; ++k1)
        val += AA[tlb::getFlatIndex(k0, k1, blocksize)] * xx[k1];
      yy[k0] -= val;
    }
  }

  template <typename bbViewType, typename yyViewType>
  KOKKOS_INLINE_FUNCTION void
  VectorCopy(const member_type &member,
             const local_ordinal_type &blocksize,
             const bbViewType &bb,
             const yyViewType &yy) const {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, blocksize), [&](const local_ordinal_type &k0) {
      yy(k0) = static_cast<typename yyViewType::const_value_type>(bb(k0));
    });
  }

  template <typename xxViewType, typename yyViewType>
  KOKKOS_INLINE_FUNCTION void
  VectorDot(const member_type &member,
            const local_ordinal_type &blocksize,
            const local_ordinal_type &lclRowID,
            const local_ordinal_type &lclColID,
            const local_ordinal_type &ii,
            const ConstUnmanaged<local_ordinal_type_1d_view> colindsub_,
            const xxViewType &xx,
            const yyViewType &yy) const {
    const size_type Aj_c  = colindsub_(lclColID);
    auto point_row_offset = A_point_rowptr(lclRowID * blocksize + ii) + Aj_c * blocksize;
    impl_scalar_type val  = 0;
    Kokkos::parallel_reduce(
        Kokkos::ThreadVectorRange(member, blocksize),
        [&](const local_ordinal_type &k1, impl_scalar_type &update) {
          update += tpetra_values(point_row_offset + k1) * xx(k1);
        },
        val);
    Kokkos::single(Kokkos::PerThread(member),
                   [=]() {
                     Kokkos::atomic_add(&yy(ii), typename yyViewType::const_value_type(-val));
                   });
  }

  // BMK: This version coalesces accesses to AA for LayoutRight blocks.
  template <typename AAViewType, typename xxViewType, typename yyViewType>
  KOKKOS_INLINE_FUNCTION void
  VectorGemv(const member_type &member,
             const local_ordinal_type &blocksize,
             const AAViewType &AA,
             const xxViewType &xx,
             const yyViewType &yy) const {
    for (local_ordinal_type k0 = 0; k0 < blocksize; ++k0) {
      impl_scalar_type val = 0;
      Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(member, blocksize),
          [&](const local_ordinal_type &k1, impl_scalar_type &update) {
            update += AA(k0, k1) * xx(k1);
          },
          val);
      Kokkos::single(Kokkos::PerThread(member),
                     [=]() {
                       Kokkos::atomic_add(&yy(k0), -val);
                     });
    }
  }

  struct SeqTag {};

  void
  operator()(const SeqTag &, const local_ordinal_type &i) const {
    const local_ordinal_type blocksize        = blocksize_requested;
    const local_ordinal_type blocksize_square = blocksize * blocksize;

    // constants
    const Kokkos::pair<local_ordinal_type, local_ordinal_type> block_range(0, blocksize);
    const local_ordinal_type num_vectors = y.extent(1);
    const local_ordinal_type row         = i * blocksize;
    for (local_ordinal_type col = 0; col < num_vectors; ++col) {
      // y := b
      impl_scalar_type *yy             = &y(row, col);
      const impl_scalar_type *const bb = &b(row, col);
      memcpy(yy, bb, sizeof(impl_scalar_type) * blocksize);

      // y -= Rx
      const size_type A_k0 = A_block_rowptr[i];
      for (size_type k = rowptr[i]; k < rowptr[i + 1]; ++k) {
        const size_type j                = A_k0 + colindsub[k];
        const impl_scalar_type *const xx = &x(A_colind[j] * blocksize, col);
        if (hasBlockCrsMatrix) {
          const impl_scalar_type *const AA = &tpetra_values(j * blocksize_square);
          SerialGemv(blocksize, AA, xx, yy);
        } else {
          for (local_ordinal_type k0 = 0; k0 < blocksize; ++k0)
            SerialDot(blocksize, i, k, k0, colindsub, xx, yy);
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const SeqTag &, const member_type &member) const {
    // constants
    const local_ordinal_type blocksize        = blocksize_requested;
    const local_ordinal_type blocksize_square = blocksize * blocksize;

    const local_ordinal_type lr = member.league_rank();
    const Kokkos::pair<local_ordinal_type, local_ordinal_type> block_range(0, blocksize);
    const local_ordinal_type num_vectors = y.extent(1);

    // subview pattern
    auto bb          = Kokkos::subview(b, block_range, 0);
    auto xx          = bb;
    auto A_block_cst = ConstUnmanaged<tpetra_block_access_view_type>(tpetra_values.data(), blocksize, blocksize);

    const local_ordinal_type row = lr * blocksize;
    for (local_ordinal_type col = 0; col < num_vectors; ++col) {
      // y := b
      auto yy = Kokkos::subview(y, Kokkos::make_pair(row, row + blocksize), col);
      bb.assign_data(&b(row, col));
      if (member.team_rank() == 0)
        VectorCopy(member, blocksize, bb, yy);
      member.team_barrier();

      // y -= Rx
      const size_type A_k0 = A_block_rowptr[lr];

      if (hasBlockCrsMatrix) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, rowptr[lr], rowptr[lr + 1]),
                             [&](const local_ordinal_type &k) {
                               const size_type j = A_k0 + colindsub[k];
                               xx.assign_data(&x(A_colind[j] * blocksize, col));
                               A_block_cst.assign_data(&tpetra_values(j * blocksize_square));
                               VectorGemv(member, blocksize, A_block_cst, xx, yy);
                             });
      } else {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, rowptr[lr], rowptr[lr + 1]),
                             [&](const local_ordinal_type &k) {
                               const size_type j = A_k0 + colindsub[k];
                               xx.assign_data(&x(A_colind[j] * blocksize, col));

                               for (local_ordinal_type k0 = 0; k0 < blocksize; ++k0)
                                 VectorDot(member, blocksize, lr, k, k0, colindsub, xx, yy);
                             });
      }
    }
  }

};

// y = b - Rx; seq method
template <typename MatrixType>
void ComputeResidualVector<MatrixType, BlockTriDiContainerDetails::ImplSimdTag>::run(const impl_scalar_type_2d_view_tpetra &y_,
                                                                                     const Const<impl_scalar_type_2d_view_tpetra> &b_,
                                                                                     const impl_scalar_type_2d_view_tpetra &x_) {
  IFPACK2_BLOCKHELPER_PROFILER_REGION_BEGIN;
  IFPACK2_BLOCKHELPER_TIMER_WITH_FENCE("BlockTriDi::ComputeResidual::<SeqTag>", ComputeResidual0, execution_space);

  using Functor = ComputeResidualFunctor<typename impl_type::matrix_type>;
  Functor functor(*this);

  functor.y = y_;
  functor.b = b_;
  functor.x = x_;
  if constexpr (is_device<execution_space>::value) {
    const local_ordinal_type blocksize   = blocksize_requested;
    const local_ordinal_type team_size   = 8;
    const local_ordinal_type vector_size = ComputeResidualVectorRecommendedVectorSize<execution_space>(blocksize, team_size);
    const Kokkos::TeamPolicy<execution_space, typename Functor::SeqTag> policy(rowptr.extent(0) - 1, team_size, vector_size);
    Kokkos::parallel_for("ComputeResidual::TeamPolicy::run<SeqTag>", policy, functor);
  } else {
    const Kokkos::RangePolicy<execution_space, typename Functor::SeqTag> policy(0, rowptr.extent(0) - 1);
    Kokkos::parallel_for("ComputeResidual::RangePolicy::run<SeqTag>", policy, functor);
  }
  IFPACK2_BLOCKHELPER_PROFILER_REGION_END;
  IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space)
}

}  // namespace Ifpack2::BlockHelperDetails

#define IFPACK2_BLOCKCOMPUTERESIDUALVECTOR_INSTANT(S, LO, GO, N) \
  template class Ifpack2::BlockHelperDetails::ComputeResidualVector<Tpetra::RowMatrix<S, LO, GO, N>>;

#endif
