// @HEADER
// *****************************************************************************
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//
// Copyright 2009 NTESS and the Ifpack2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef IFPACK2_BLOCKCOMPUTERESIDUAL_PACKED_HPP
#define IFPACK2_BLOCKCOMPUTERESIDUAL_PACKED_HPP

// Packed (SIMD-lane-interleaved) representation of R = A - tridiag(A), and
// the kernels that use it: residual, fused block-Jacobi residual+solve, and
// conversion of multivectors between Tpetra and packed layouts.
//
// Packed layout: rows of the parts (lines) are grouped in packs of
// vector_length parts. Pack row pri holds row j of each of the (up to)
// vector_length parts of a pack, and lane v (innermost dimension) selects the
// part. All packed multivectors here have layout (npackrows, blocksize,
// num_vectors, vector_length), the same as the solver's packed work vector
// (vector_type (npackrows, blocksize, num_vectors)).
//
// R is stored as a "packed CSR" over pack rows: pack row pri owns slots
// rowptr(pri) ... rowptr(pri+1)-1. A slot holds one blocksize x blocksize
// block for each lane, plus the location of the x entries that block
// multiplies. The number of slots in a pack row is the max over its lanes of
// the number of R blocks in that row; lanes with fewer blocks are padded with
// zero blocks and an "absent" x index.

#include "Ifpack2_BlockHelper.hpp"

namespace Ifpack2 {

namespace BlockHelperDetails {

// The block sizes for which the packed kernels are specialized at compile time.
#define IFPACK2_BLOCKHELPER_PACKED_BLOCKSIZE_SWITCH(blocksize, CASE_MACRO) \
  switch (blocksize) {                                                     \
    case 2: CASE_MACRO(2);                                                 \
    case 3: CASE_MACRO(3);                                                 \
    case 4: CASE_MACRO(4);                                                 \
    case 5: CASE_MACRO(5);                                                 \
    case 6: CASE_MACRO(6);                                                 \
    case 7: CASE_MACRO(7);                                                 \
    case 8: CASE_MACRO(8);                                                 \
    case 9: CASE_MACRO(9);                                                 \
    case 10: CASE_MACRO(10);                                               \
    case 11: CASE_MACRO(11);                                               \
    case 12: CASE_MACRO(12);                                               \
    case 13: CASE_MACRO(13);                                               \
    case 14: CASE_MACRO(14);                                               \
    case 15: CASE_MACRO(15);                                               \
    case 16: CASE_MACRO(16);                                               \
    case 17: CASE_MACRO(17);                                               \
    case 18: CASE_MACRO(18);                                               \
    case 19: CASE_MACRO(19);                                               \
    default: CASE_MACRO(0);                                                \
  }

template <typename MatrixType>
struct PackedAmD {
  using impl_type          = ImplType<MatrixType>;
  using device_type        = typename impl_type::device_type;
  using local_ordinal_type = typename impl_type::local_ordinal_type;
  using impl_scalar_type   = typename impl_type::impl_scalar_type;
  using size_type_1d_view  = typename impl_type::size_type_1d_view;

  using lo_2d_view     = Kokkos::View<local_ordinal_type **, Kokkos::LayoutRight, device_type>;
  using i64_2d_view    = Kokkos::View<int64_t **, Kokkos::LayoutRight, device_type>;
  using values_view    = Kokkos::View<impl_scalar_type ****, Kokkos::LayoutRight, device_type>;
  using values_1d_view = Kokkos::View<impl_scalar_type *, device_type>;

  // Special values of xidx
  static constexpr local_ordinal_type absent = -1;
  // remote column with remote LID r is encoded as -(r + 2)
  KOKKOS_INLINE_FUNCTION static constexpr local_ordinal_type encodeRemote(const local_ordinal_type r) { return -(r + 2); }
  KOKKOS_INLINE_FUNCTION static constexpr local_ordinal_type decodeRemote(const local_ordinal_type x) { return -x - 2; }

  struct Graph {
    // (npackrows + 1): slot pointers
    size_type_1d_view rowptr;
    // (nslots, vector_length): packed source of x for each slot and lane:
    // >= 0: packed local row index (pri * vector_length + v) of x
    // == absent: no block here (padding)
    // <= -2: remote, decodeRemote gives the row of the remote multivector
    lo_2d_view xidx;
    // (nslots, vector_length): source of the block in A's values
    // (BlockCrsMatrix: offset of the block; CrsMatrix: block column offset within the point rows).
    // -1 if absent.
    i64_2d_view src;
    // (nslots, vector_length): CrsMatrix only: local block row of the source block.
    lo_2d_view src_row;
    // (nslots, blocksize, blocksize, vector_length): values(slot, k1, k0, lane) is
    // entry (k0, k1) of the block (i.e. the packed blocks are column-major)
    values_view values;
    // Number of slots
    size_t nslots = 0;
  };

  // If the container overlaps communication and computation, owned holds the
  // blocks multiplying owned columns and remote the ones multiplying remote
  // columns. Otherwise, owned holds all of R and remote is empty.
  Graph owned, remote;

  // (number of local rows): row LID -> packed row index (pri * vector_length + v)
  Kokkos::View<local_ordinal_type *, device_type> lid2packed;

  local_ordinal_type blocksize      = 0;
  local_ordinal_type num_pack_rows  = 0;
  bool is_block_crs                 = true;
  bool overlap                      = false;

  PackedAmD()                   = default;
  PackedAmD(const PackedAmD &b) = default;
};

///
/// Symbolic phase: build the packed R graph(s).
///
/// R_rowptr/R_colindsub: per local row (LID) lists of indices (into A's local
/// row) of the blocks in R, as in AmD. col2row maps column LID -> row LID of
/// owned columns (-1 if remote) and lclrow2idx maps row LID -> row index.
///
template <typename MatrixType, typename LocalRowptr, typename LocalColidx, typename LOView>
void buildPackedAmDGraph(typename PackedAmD<MatrixType>::Graph &graph,
                         const PartInterface<MatrixType> &interf,
                         const typename ImplType<MatrixType>::size_type_1d_view &R_rowptr,
                         const typename ImplType<MatrixType>::local_ordinal_type_1d_view &R_colindsub,
                         const LocalRowptr &A_rowptr,
                         const LocalColidx &A_colidx,
                         const LOView &col2row,
                         const LOView &lclrow2idx,
                         const typename ImplType<MatrixType>::local_ordinal_type num_pack_rows,
                         const typename ImplType<MatrixType>::local_ordinal_type blocksize,
                         const bool is_block_crs) {
  using impl_type          = ImplType<MatrixType>;
  using execution_space    = typename impl_type::execution_space;
  using local_ordinal_type = typename impl_type::local_ordinal_type;
  using size_type          = typename impl_type::size_type;
  using packed_amd_type    = PackedAmD<MatrixType>;
  using size_type_1d_view  = typename impl_type::size_type_1d_view;
  using lo_2d_view         = typename packed_amd_type::lo_2d_view;
  using i64_2d_view        = typename packed_amd_type::i64_2d_view;
  using values_view        = typename packed_amd_type::values_view;
  constexpr int vl         = impl_type::vector_length;

  const local_ordinal_type nrows = interf.lclrow.extent(0);
  const auto lclrow              = interf.lclrow;
  const auto rowidx2part         = interf.rowidx2part;
  const auto partptr             = interf.partptr;
  const auto part2packrowidx0    = interf.part2packrowidx0;

  // Number of slots in each pack row: max over lanes of the number of blocks.
  size_type_1d_view rowptr("packed_amd.rowptr", num_pack_rows + 1);
  if (R_rowptr.extent(0) > 0) {
    Kokkos::parallel_for(
        "buildPackedAmDGraph::count", Kokkos::RangePolicy<execution_space>(0, nrows),
        KOKKOS_LAMBDA(const local_ordinal_type ri) {
          const local_ordinal_type lr      = lclrow(ri);
          const local_ordinal_type partidx = rowidx2part(ri);
          const local_ordinal_type pri     = part2packrowidx0(partidx) + (ri - partptr(partidx));
          const size_type n                = R_rowptr(lr + 1) - R_rowptr(lr);
          Kokkos::atomic_max(&rowptr(pri), n);
        });
  }
  size_type nslots = 0;
  Kokkos::parallel_scan(
      "buildPackedAmDGraph::scan", Kokkos::RangePolicy<execution_space>(0, num_pack_rows + 1),
      KOKKOS_LAMBDA(const local_ordinal_type i, size_type &update, const bool final) {
        const size_type n = rowptr(i);
        if (final) rowptr(i) = update;
        update += n;
      },
      nslots);

  graph.rowptr  = rowptr;
  graph.nslots  = nslots;
  graph.xidx    = lo_2d_view(do_not_initialize_tag("packed_amd.xidx"), nslots, vl);
  graph.src     = i64_2d_view(do_not_initialize_tag("packed_amd.src"), nslots, vl);
  graph.src_row = is_block_crs ? lo_2d_view() : lo_2d_view(do_not_initialize_tag("packed_amd.src_row"), nslots, vl);
  graph.values  = values_view(do_not_initialize_tag("packed_amd.values"), nslots, blocksize, blocksize, vl);
  Kokkos::deep_copy(graph.xidx, packed_amd_type::absent);
  Kokkos::deep_copy(graph.src, int64_t(-1));
  if (!is_block_crs) Kokkos::deep_copy(graph.src_row, 0);

  if (nslots == 0) return;

  const auto xidx    = graph.xidx;
  const auto src     = graph.src;
  const auto src_row = graph.src_row;
  const int64_t bs   = blocksize;
  Kokkos::parallel_for(
      "buildPackedAmDGraph::fill", Kokkos::RangePolicy<execution_space>(0, nrows),
      KOKKOS_LAMBDA(const local_ordinal_type ri) {
        const local_ordinal_type lr      = lclrow(ri);
        const local_ordinal_type partidx = rowidx2part(ri);
        const local_ordinal_type pri     = part2packrowidx0(partidx) + (ri - partptr(partidx));
        const local_ordinal_type v       = partidx % vl;
        const size_type slot0            = rowptr(pri);
        const size_type j0               = A_rowptr(lr);
        for (size_type k = R_rowptr(lr); k < R_rowptr(lr + 1); ++k) {
          const size_type slot          = slot0 + (k - R_rowptr(lr));
          const local_ordinal_type entry = R_colindsub(k);
          const size_type j             = j0 + entry;
          const local_ordinal_type lc   = A_colidx(j);
          const local_ordinal_type lc2r = lc < (local_ordinal_type)col2row.extent(0) ? col2row(lc) : local_ordinal_type(-1);
          if (lc2r != local_ordinal_type(-1)) {
            const local_ordinal_type ri2 = lclrow2idx(lc2r);
            const local_ordinal_type p2  = rowidx2part(ri2);
            xidx(slot, v)                = (part2packrowidx0(p2) + (ri2 - partptr(p2))) * vl + (p2 % vl);
          } else {
            xidx(slot, v) = packed_amd_type::encodeRemote(lc - nrows);
          }
          if (is_block_crs) {
            src(slot, v) = int64_t(j) * bs * bs;
          } else {
            src(slot, v)     = int64_t(entry) * bs;
            src_row(slot, v) = lr;
          }
        }
      });
}

///
/// Numeric phase: gather the values of R from A into the packed graph.
///
/// The packed blocks are column-major: values(slot, k1, k0, lane) = A(k0, k1).
/// A team handles slots_per_team slots: it reads the blocks of each lane
/// (contiguous in A for a BlockCrsMatrix) into scratch, then writes them out
/// transposed and interleaved, so both the reads and the writes are coalesced.
///
template <typename MatrixType, int B, typename AValues, typename APointRowptr>
struct FillPackedAmDFunctor {
  using impl_type          = ImplType<MatrixType>;
  using execution_space    = typename impl_type::execution_space;
  using impl_scalar_type   = typename impl_type::impl_scalar_type;
  using local_ordinal_type = typename impl_type::local_ordinal_type;
  using packed_amd_type    = PackedAmD<MatrixType>;
  using member_type        = typename Kokkos::TeamPolicy<execution_space>::member_type;
  static constexpr int vl  = impl_type::vector_length;

  Unmanaged<typename packed_amd_type::values_view> values;
  ConstUnmanaged<typename packed_amd_type::i64_2d_view> src;
  ConstUnmanaged<typename packed_amd_type::lo_2d_view> src_row;
  AValues A_values;
  APointRowptr A_point_rowptr;
  int blocksize_requested;
  int slots_per_team;
  int64_t nslots;
  int scratch_level;
  bool is_block_crs;

  KOKKOS_INLINE_FUNCTION static size_t scratchSize(const int blocksize, const int slots_per_team) {
    return sizeof(impl_scalar_type) * size_t(blocksize) * blocksize * vl * slots_per_team;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type &member) const {
    const int bs         = (B == 0 ? blocksize_requested : B);
    const int bs2        = bs * bs;
    const int slot_size  = bs2 * vl;
    const int64_t slot0  = int64_t(member.league_rank()) * slots_per_team;
    const int nitems     = int(Kokkos::min(int64_t(slots_per_team), nslots - slot0)) * slot_size;
    impl_scalar_type *sm = reinterpret_cast<impl_scalar_type *>(member.team_scratch(scratch_level).get_shmem(scratchSize(bs, slots_per_team)));
    // read: t -> (slot, lane, k0, k1) with k1 fastest (A's row-major block)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, nitems), [&](const int t) {
      const int ls         = t / slot_size;
      const int r          = t - ls * slot_size;
      const int v          = r / bs2;
      const int e          = r - v * bs2;
      const int k0         = e / bs;
      const int k1         = e - k0 * bs;
      const int64_t slot   = slot0 + ls;
      const int64_t sv     = src(slot, v);
      impl_scalar_type a(0);
      if (sv >= 0) {
        if (is_block_crs)
          a = A_values(sv + e);
        else
          a = A_values(A_point_rowptr(int64_t(src_row(slot, v)) * bs + k0) + sv + k1);
      }
      sm[ls * slot_size + (k1 * bs + k0) * vl + v] = a;
    });
    member.team_barrier();
    // write: contiguous
    impl_scalar_type *out = values.data() + slot0 * slot_size;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, nitems), [&](const int t) {
      out[t] = sm[t];
    });
  }
};

template <typename MatrixType, typename AValues, typename APointRowptr>
void fillPackedAmDValues(const typename PackedAmD<MatrixType>::Graph &graph,
                         const AValues &A_values,
                         const APointRowptr &A_point_rowptr,
                         const typename ImplType<MatrixType>::local_ordinal_type blocksize,
                         const bool is_block_crs,
                         const typename ImplType<MatrixType>::execution_space &exec) {
  using impl_type        = ImplType<MatrixType>;
  using execution_space  = typename impl_type::execution_space;
  using team_policy_type = Kokkos::TeamPolicy<execution_space>;
  constexpr int vl       = impl_type::vector_length;
  if (graph.nslots == 0) return;
  static_assert(std::is_same<typename Tpetra::Impl::BlockCrsMatrixLittleBlockArrayLayout, Kokkos::LayoutRight>::value,
                "fillPackedAmDValues assumes row-major blocks in BlockCrsMatrix");

  const int slot_size      = blocksize * blocksize * vl;
  const int slots_per_team = Kokkos::max(1, 2048 / slot_size);
  const int64_t nteams     = (int64_t(graph.nslots) + slots_per_team - 1) / slots_per_team;
  const int team_size      = is_device<execution_space>::value ? 256 : 1;
#define IFPACK2_PACKED_FILL_CASE(B)                                                                                 \
  {                                                                                                                 \
    using functor_type        = FillPackedAmDFunctor<MatrixType, B, AValues, APointRowptr>;                         \
    const size_t scratch_size = functor_type::scratchSize(blocksize, slots_per_team);                               \
    const int scratch_level   = (scratch_size <= size_t(team_policy_type::scratch_size_max(0))) ? 0 : 1;            \
    functor_type f{graph.values, graph.src, graph.src_row, A_values, A_point_rowptr, blocksize, slots_per_team,     \
                   int64_t(graph.nslots), scratch_level, is_block_crs};                                             \
    team_policy_type policy(exec, nteams, team_size);                                                               \
    policy.set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_size));                                          \
    Kokkos::parallel_for("fillPackedAmDValues", policy, f);                                                         \
  }                                                                                                                 \
  break
  IFPACK2_BLOCKHELPER_PACKED_BLOCKSIZE_SWITCH(blocksize, IFPACK2_PACKED_FILL_CASE)
#undef IFPACK2_PACKED_FILL_CASE
}

///
/// Conversion between Tpetra (LayoutLeft, row LID major) and packed multivectors.
///
template <typename MatrixType>
struct PackedMultiVector {
  using impl_type                       = ImplType<MatrixType>;
  using execution_space                 = typename impl_type::execution_space;
  using local_ordinal_type              = typename impl_type::local_ordinal_type;
  using impl_scalar_type                = typename impl_type::impl_scalar_type;
  using device_type                     = typename impl_type::device_type;
  using view_type                       = Kokkos::View<impl_scalar_type ****, Kokkos::LayoutRight, device_type>;
  using impl_scalar_type_2d_view_tpetra = typename impl_type::impl_scalar_type_2d_view_tpetra;
  static constexpr int vl               = impl_type::vector_length;

  // packed(lid2packed(lr)) = tpetra(lr)
  static void pack(const view_type &packed,
                   const Kokkos::View<const impl_scalar_type **, Kokkos::LayoutLeft, typename impl_type::node_device_type> &mv,
                   const Kokkos::View<local_ordinal_type *, device_type> &lid2packed) {
    IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::PackMultiVector", PackMultiVector);
    const local_ordinal_type bs   = packed.extent(1);
    const local_ordinal_type nvec = packed.extent(2);
    const local_ordinal_type n    = mv.extent(0);
    Kokkos::parallel_for(
        "PackedMultiVector::pack", Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(const local_ordinal_type i) {
          const local_ordinal_type lr = i / bs;
          const local_ordinal_type k  = i % bs;
          const local_ordinal_type p  = lid2packed(lr);
          for (local_ordinal_type col = 0; col < nvec; ++col)
            packed(p / vl, k, col, p % vl) = mv(i, col);
        });
    IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space)
  }

  // tpetra(lr) = packed(lid2packed(lr))
  static void unpack(const impl_scalar_type_2d_view_tpetra &mv,
                     const view_type &packed,
                     const Kokkos::View<local_ordinal_type *, device_type> &lid2packed) {
    IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::UnpackMultiVector", UnpackMultiVector);
    const local_ordinal_type bs   = packed.extent(1);
    const local_ordinal_type nvec = packed.extent(2);
    const local_ordinal_type n    = mv.extent(0);
    Kokkos::parallel_for(
        "PackedMultiVector::unpack", Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(const local_ordinal_type i) {
          const local_ordinal_type lr = i / bs;
          const local_ordinal_type k  = i % bs;
          const local_ordinal_type p  = lid2packed(lr);
          for (local_ordinal_type col = 0; col < nvec; ++col)
            mv(i, col) = packed(p / vl, k, col, p % vl);
        });
    IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space)
  }
};

///
/// Packed residual kernel:
///   out = b - R x     (Init)
///   out = out - R x   (Accum)
/// One thread per (pack row, block row k0, lane), lane fastest. With the
/// column-major packed blocks of R, the threads of a warp read contiguous
/// memory for each block column k1.
///
template <typename MatrixType, int B, bool Init>
struct PackedResidualFunctor {
  using impl_type          = ImplType<MatrixType>;
  using device_type        = typename impl_type::device_type;
  using local_ordinal_type = typename impl_type::local_ordinal_type;
  using size_type          = typename impl_type::size_type;
  using impl_scalar_type   = typename impl_type::impl_scalar_type;
  using btdm_scalar_type   = typename impl_type::btdm_scalar_type;
  using packed_amd_type    = PackedAmD<MatrixType>;
  static constexpr int vl  = impl_type::vector_length;

  using const_packed_mv_type = Kokkos::View<const impl_scalar_type ****, Kokkos::LayoutRight, device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using btdm_mv_type         = Kokkos::View<btdm_scalar_type ****, Kokkos::LayoutRight, device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using remote_mv_type       = ConstUnmanaged<typename impl_type::impl_scalar_type_2d_view_tpetra>;

  ConstUnmanaged<typename impl_type::size_type_1d_view> rowptr;
  Kokkos::View<const local_ordinal_type **, Kokkos::LayoutRight, device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>> xidx;
  const_packed_mv_type R;
  const_packed_mv_type b, x;
  remote_mv_type x_remote;
  btdm_mv_type out;
  local_ordinal_type blocksize_requested;
  local_ordinal_type num_vectors;

  KOKKOS_INLINE_FUNCTION
  void operator()(const local_ordinal_type i) const {
    const local_ordinal_type blocksize = (B == 0 ? blocksize_requested : B);
    const local_ordinal_type v         = i % vl;
    const local_ordinal_type k0        = (i / vl) % blocksize;
    const local_ordinal_type pri       = i / (vl * blocksize);
    const size_type s_beg              = rowptr(pri);
    const size_type s_end              = rowptr(pri + 1);
    for (local_ordinal_type col = 0; col < num_vectors; ++col) {
      impl_scalar_type acc = Init ? b(pri, k0, col, v) : impl_scalar_type(out(pri, k0, col, v));
      for (size_type s = s_beg; s < s_end; ++s) {
        const local_ordinal_type xi = xidx(s, v);
        if (xi == packed_amd_type::absent) continue;
        impl_scalar_type val = 0;
        if (xi >= 0) {
          const local_ordinal_type xpri = xi / vl;
          const local_ordinal_type xv   = xi % vl;
          for (local_ordinal_type k1 = 0; k1 < blocksize; ++k1)
            val += R(s, k1, k0, v) * x(xpri, k1, col, xv);
        } else {
          const local_ordinal_type r = packed_amd_type::decodeRemote(xi) * blocksize;
          for (local_ordinal_type k1 = 0; k1 < blocksize; ++k1)
            val += R(s, k1, k0, v) * x_remote(r + k1, col);
        }
        acc -= val;
      }
      out(pri, k0, col, v) = acc;
    }
  }
};

///
/// Fused block Jacobi sweep (all parts have one row; D^-1 is stored in the
/// packed tridiag values, one block per pack row):
///   YZero: y_new = df * D^-1 b                       (y = 0)
///   Init:  y_new = y + df * (D^-1 (b - R x) - y)
///   Accum: y_new = y + df * (D^-1 (out - R x) - y)
/// where x is y and/or the remote part of y. W(team) gets the squared norm of
/// the update (as if df = 1) for the pack rows of the team.
///
/// A team handles rows_per_team pack rows. Each thread computes the residual
/// for a (pack row, k0, lane) into scratch, then applies D^-1.
///
enum class PackedJacobiMode { YZero,
                              Init,
                              Accum };

template <typename MatrixType, int B, PackedJacobiMode Mode>
struct PackedJacobiFunctor {
  using impl_type          = ImplType<MatrixType>;
  using execution_space    = typename impl_type::execution_space;
  using device_type        = typename impl_type::device_type;
  using local_ordinal_type = typename impl_type::local_ordinal_type;
  using size_type          = typename impl_type::size_type;
  using impl_scalar_type   = typename impl_type::impl_scalar_type;
  using magnitude_type     = typename impl_type::magnitude_type;
  using btdm_scalar_type   = typename impl_type::btdm_scalar_type;
  using packed_amd_type    = PackedAmD<MatrixType>;
  using member_type        = typename Kokkos::TeamPolicy<execution_space>::member_type;
  static constexpr int vl  = impl_type::vector_length;

  using packed_mv_type       = Kokkos::View<impl_scalar_type ****, Kokkos::LayoutRight, device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using const_packed_mv_type = Kokkos::View<const impl_scalar_type ****, Kokkos::LayoutRight, device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using const_btdm_mv_type   = Kokkos::View<const btdm_scalar_type ****, Kokkos::LayoutRight, device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using remote_mv_type       = ConstUnmanaged<typename impl_type::impl_scalar_type_2d_view_tpetra>;

  ConstUnmanaged<typename impl_type::size_type_1d_view> rowptr;
  Kokkos::View<const local_ordinal_type **, Kokkos::LayoutRight, device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>> xidx;
  const_packed_mv_type R;
  const_packed_mv_type b, x;
  remote_mv_type x_remote;
  const_btdm_mv_type in;  // Accum: partial residual
  const_btdm_mv_type Dinv;
  packed_mv_type y_new;
  Unmanaged<typename impl_type::impl_scalar_type_1d_view> W;
  impl_scalar_type df;
  local_ordinal_type blocksize_requested;
  local_ordinal_type num_vectors;
  local_ordinal_type num_pack_rows;
  local_ordinal_type rows_per_team;

  KOKKOS_INLINE_FUNCTION static size_t scratchSize(const local_ordinal_type blocksize, const local_ordinal_type rows_per_team) {
    return sizeof(impl_scalar_type) * size_t(rows_per_team) * blocksize * vl;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type &member) const {
    const local_ordinal_type blocksize = (B == 0 ? blocksize_requested : B);
    const local_ordinal_type row_size  = blocksize * vl;
    const local_ordinal_type pri0      = member.league_rank() * rows_per_team;
    const local_ordinal_type nitems    = Kokkos::min(rows_per_team, num_pack_rows - pri0) * row_size;
    impl_scalar_type *r                = reinterpret_cast<impl_scalar_type *>(member.team_scratch(0).get_shmem(scratchSize(blocksize, rows_per_team)));

    magnitude_type norm = 0;
    for (local_ordinal_type col = 0; col < num_vectors; ++col) {
      if (col) member.team_barrier();
      // r = b - R x (or in - R x, or b)
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, nitems), [&](const local_ordinal_type t) {
        const local_ordinal_type v   = t % vl;
        const local_ordinal_type k0  = (t / vl) % blocksize;
        const local_ordinal_type pri = pri0 + t / row_size;
        impl_scalar_type acc         = (Mode == PackedJacobiMode::Accum) ? impl_scalar_type(in(pri, k0, col, v)) : b(pri, k0, col, v);
        if constexpr (Mode != PackedJacobiMode::YZero) {
          const size_type s_end = rowptr(pri + 1);
          for (size_type s = rowptr(pri); s < s_end; ++s) {
            const local_ordinal_type xi = xidx(s, v);
            if (xi == packed_amd_type::absent) continue;
            impl_scalar_type val = 0;
            if (xi >= 0) {
              const local_ordinal_type xpri = xi / vl;
              const local_ordinal_type xv   = xi % vl;
              for (local_ordinal_type k1 = 0; k1 < blocksize; ++k1)
                val += R(s, k1, k0, v) * x(xpri, k1, col, xv);
            } else {
              const local_ordinal_type rr = packed_amd_type::decodeRemote(xi) * blocksize;
              for (local_ordinal_type k1 = 0; k1 < blocksize; ++k1)
                val += R(s, k1, k0, v) * x_remote(rr + k1, col);
            }
            acc -= val;
          }
        }
        r[t] = acc;
      });
      member.team_barrier();
      // y_new = y + df (D^-1 r - y)
      magnitude_type col_norm = 0;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(member, nitems),
          [&](const local_ordinal_type t, magnitude_type &update) {
            const local_ordinal_type v      = t % vl;
            const local_ordinal_type k0     = (t / vl) % blocksize;
            const local_ordinal_type p      = t / row_size;
            const local_ordinal_type pri    = pri0 + p;
            const impl_scalar_type *const rp = r + p * row_size + v;
            impl_scalar_type z               = 0;
            for (local_ordinal_type k1 = 0; k1 < blocksize; ++k1)
              z += impl_scalar_type(Dinv(pri, k0, k1, v)) * rp[k1 * vl];
            const impl_scalar_type y_old = (Mode == PackedJacobiMode::YZero) ? impl_scalar_type(0) : x(pri, k0, col, v);
            const impl_scalar_type yd    = z - y_old;
            const magnitude_type yd_abs  = KokkosKernels::ArithTraits<impl_scalar_type>::abs(yd);
            update += yd_abs * yd_abs;
            y_new(pri, k0, col, v) = y_old + df * yd;
          },
          col_norm);
      norm += col_norm;
    }
    Kokkos::single(Kokkos::PerTeam(member), [&]() { W(member.league_rank()) = norm; });
  }
};

///
/// Driver for the packed kernels.
///
template <typename MatrixType>
struct PackedResidual {
  using impl_type                = ImplType<MatrixType>;
  using execution_space          = typename impl_type::execution_space;
  using device_type              = typename impl_type::device_type;
  using local_ordinal_type       = typename impl_type::local_ordinal_type;
  using impl_scalar_type         = typename impl_type::impl_scalar_type;
  using btdm_scalar_type         = typename impl_type::btdm_scalar_type;
  using impl_scalar_type_1d_view = typename impl_type::impl_scalar_type_1d_view;
  using packed_amd_type          = PackedAmD<MatrixType>;
  using graph_type               = typename packed_amd_type::Graph;
  using packed_mv_type           = typename PackedMultiVector<MatrixType>::view_type;
  using btdm_mv_type             = Kokkos::View<btdm_scalar_type ****, Kokkos::LayoutRight, device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using remote_mv_type           = typename impl_type::impl_scalar_type_2d_view_tpetra;
  static constexpr int vl        = impl_type::vector_length;

  // out = b - R x   (init = true)  or   out -= R x   (init = false)
  static void residual(const graph_type &graph,
                       const packed_mv_type &b,
                       const packed_mv_type &x,
                       const remote_mv_type &x_remote,
                       const btdm_mv_type &out,
                       const local_ordinal_type blocksize,
                       const bool init) {
    IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::PackedResidual", PackedResidual0);
    const local_ordinal_type n = out.extent(0) * blocksize * vl;
#define IFPACK2_PACKED_RESIDUAL_CASE(B)                                                                    \
  {                                                                                                        \
    if (init) {                                                                                            \
      PackedResidualFunctor<MatrixType, B, true> f{graph.rowptr, graph.xidx, graph.values, b, x, x_remote, \
                                                   out, blocksize, local_ordinal_type(out.extent(2))};     \
      Kokkos::parallel_for("BlockTriDi::PackedResidual::Init",                                             \
                           Kokkos::RangePolicy<execution_space>(0, n), f);                                 \
    } else {                                                                                               \
      PackedResidualFunctor<MatrixType, B, false> f{graph.rowptr, graph.xidx, graph.values, b, x, x_remote, \
                                                    out, blocksize, local_ordinal_type(out.extent(2))};     \
      Kokkos::parallel_for("BlockTriDi::PackedResidual::Accum",                                            \
                           Kokkos::RangePolicy<execution_space>(0, n), f);                                 \
    }                                                                                                      \
  }                                                                                                        \
  break
    IFPACK2_BLOCKHELPER_PACKED_BLOCKSIZE_SWITCH(blocksize, IFPACK2_PACKED_RESIDUAL_CASE)
#undef IFPACK2_PACKED_RESIDUAL_CASE
    IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space)
  }

  // Number of pack rows per team in the fused Jacobi kernel
  static local_ordinal_type jacobiRowsPerTeam(const local_ordinal_type blocksize) {
    return Kokkos::max(local_ordinal_type(1), local_ordinal_type(256 / (blocksize * vl)));
  }
  // Size of W needed by the fused Jacobi kernel
  static local_ordinal_type jacobiNumTeams(const local_ordinal_type num_pack_rows, const local_ordinal_type blocksize) {
    const local_ordinal_type rpt = jacobiRowsPerTeam(blocksize);
    return (num_pack_rows + rpt - 1) / rpt;
  }

  // Fused block Jacobi sweep.
  static void jacobi(const PackedJacobiMode mode,
                     const graph_type &graph,
                     const packed_mv_type &b,
                     const packed_mv_type &x,
                     const remote_mv_type &x_remote,
                     const btdm_mv_type &in,
                     const btdm_mv_type &Dinv,
                     const packed_mv_type &y_new,
                     const impl_scalar_type_1d_view &W,
                     const impl_scalar_type df,
                     const local_ordinal_type blocksize) {
    IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::PackedJacobi", PackedJacobi0);
    const local_ordinal_type num_pack_rows = y_new.extent(0);
    const local_ordinal_type rpt           = jacobiRowsPerTeam(blocksize);
    const local_ordinal_type nteams        = jacobiNumTeams(num_pack_rows, blocksize);
    const local_ordinal_type nitems        = rpt * blocksize * vl;
    local_ordinal_type team_size           = 1;
    if constexpr (is_device<execution_space>::value) {
      team_size = Kokkos::min(local_ordinal_type(256), ((nitems + 31) / 32) * 32);
    }
#define IFPACK2_PACKED_JACOBI_LAUNCH(B, MODE, LABEL)                                                                  \
  {                                                                                                                   \
    using functor_type = PackedJacobiFunctor<MatrixType, B, MODE>;                                                    \
    functor_type f{graph.rowptr, graph.xidx, graph.values, b, x, x_remote, in, Dinv, y_new, W, df, blocksize,         \
                   local_ordinal_type(y_new.extent(2)), num_pack_rows, rpt};                                          \
    Kokkos::TeamPolicy<execution_space> policy(nteams, team_size);                                                    \
    policy.set_scratch_size(0, Kokkos::PerTeam(functor_type::scratchSize(blocksize, rpt)));                           \
    Kokkos::parallel_for(LABEL, policy, f);                                                                           \
  }
#define IFPACK2_PACKED_JACOBI_CASE(B)                                                                          \
  if (mode == PackedJacobiMode::YZero)                                                                         \
    IFPACK2_PACKED_JACOBI_LAUNCH(B, PackedJacobiMode::YZero, "BlockTriDi::PackedJacobi::YZero")                \
  else if (mode == PackedJacobiMode::Init)                                                                     \
    IFPACK2_PACKED_JACOBI_LAUNCH(B, PackedJacobiMode::Init, "BlockTriDi::PackedJacobi::Init")                  \
  else                                                                                                         \
    IFPACK2_PACKED_JACOBI_LAUNCH(B, PackedJacobiMode::Accum, "BlockTriDi::PackedJacobi::Accum")                \
  break
    IFPACK2_BLOCKHELPER_PACKED_BLOCKSIZE_SWITCH(blocksize, IFPACK2_PACKED_JACOBI_CASE)
#undef IFPACK2_PACKED_JACOBI_CASE
#undef IFPACK2_PACKED_JACOBI_LAUNCH
    IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space)
  }
};

// Find the offset (in scalars) of the diagonal block of each block row of a BlockCrsMatrix.
// Used by the fused block Jacobi factorization.
template <typename ExecSpace, typename DiagOffsets, typename Rowptrs,
          typename Entries>
DiagOffsets findDiagOffsets(const Rowptrs& rowptrs, const Entries& entries,
                            size_t nrows, int blocksize) {
  DiagOffsets diag_offsets(do_not_initialize_tag("btdm.diag_offsets"), nrows);
  int err1 = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<ExecSpace>(0, nrows),
      KOKKOS_LAMBDA(size_t row, int& err2) {
        auto rowBegin = rowptrs(row);
        auto rowEnd   = rowptrs(row + 1);
        for (size_t j = rowBegin; j < rowEnd; j++) {
          if (size_t(entries(j)) == row) {
            diag_offsets(row) = j * blocksize * blocksize;
            return;
          }
        }
        err2++;
      },
      err1);
  TEUCHOS_TEST_FOR_EXCEPT_MSG(
      err1, "Ifpack2 BTD: at least one row has no diagonal entry");
  return diag_offsets;
}

}  // namespace BlockHelperDetails

}  // namespace Ifpack2

#endif
