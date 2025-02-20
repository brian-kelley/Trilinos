// @HEADER
// *****************************************************************************
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//
// Copyright 2009 NTESS and the Ifpack2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef IFPACK2_BLOCKCOMPUTERESAND_SOLVE_IMPL_HPP
#define IFPACK2_BLOCKCOMPUTERESAND_SOLVE_IMPL_HPP

#include "Ifpack2_BlockHelper.hpp"
#include "Ifpack2_BlockComputeResidualVector.hpp"

namespace Ifpack2::BlockHelperDetails {

  template<typename MatrixType>
  struct ComputeResidualAndSolve {
  public:
    using impl_type = BlockHelperDetails::ImplType<MatrixType>;
    using node_device_type = typename impl_type::node_device_type;
    using execution_space = typename impl_type::execution_space;
    using memory_space = typename impl_type::memory_space;

    using local_ordinal_type = typename impl_type::local_ordinal_type;
    using size_type = typename impl_type::size_type;
    using impl_scalar_type = typename impl_type::impl_scalar_type;
    using magnitude_type = typename impl_type::magnitude_type;
    using btdm_scalar_type = typename impl_type::btdm_scalar_type;
    using btdm_magnitude_type = typename impl_type::btdm_magnitude_type;
    /// views
    using local_ordinal_type_1d_view = typename impl_type::local_ordinal_type_1d_view;
    using size_type_1d_view = typename impl_type::size_type_1d_view;
    using tpetra_block_access_view_type = typename impl_type::tpetra_block_access_view_type; // block crs (layout right)
    using impl_scalar_type_1d_view = typename impl_type::impl_scalar_type_1d_view;
    using impl_scalar_type_2d_view_tpetra = typename impl_type::impl_scalar_type_2d_view_tpetra; // block multivector (layout left)
    using vector_type_3d_view = typename impl_type::vector_type_3d_view;
    using btdm_scalar_type_4d_view = typename impl_type::btdm_scalar_type_4d_view;
    using i64_3d_view  = typename impl_type::i64_3d_view;
    static constexpr int vector_length = impl_type::vector_length;

    /// team policy member type (used in cuda)
    using member_type = typename Kokkos::TeamPolicy<execution_space>::member_type;

    // enum for max blocksize and vector length
    enum : int { max_blocksize = 32 };

    template<int B, int mode>
    struct GeneralTag
    {
      static_assert(mode >= 0 && mode <= 2,
          "BlockComputeResidualAndSolve, GeneralTag: requires 0 <= mode <= 2");
    };

    // Tag for when only a single pass does the whole residual and solve.
    // This applies to the "async" cases, and single-rank problems.
    //
    // partial_residual isn't used. This returns the local squared y-update norm.
    template<int B>
    using SinglePassTag = GeneralTag<B, 0>{};

    // Tag for doing the first stage (owned columns) of a 2-pass residual.
    // The result is saved into partial_residual in this case.
    // The inverse diagonal isn't used yet, and this doesn't produce a partial norm either.
    template<int B>
    using TwoPassOwnedTag = GeneralTag<B, 1>{};

    // Tag for doing the second stage (nonowned columns) of a 2-pass residual.
    // The results are summed with those from the first stage (partial_residual),
    // and then the inverse diagonal is applied. The local squared y-update norm is 
    template<int B>
    using TwoPassNonownedTag = GeneralTag<B, 2>{};

  private:
    ConstUnmanaged<impl_scalar_type_2d_view_tpetra> b;
    ConstUnmanaged<impl_scalar_type_2d_view_tpetra> x; // x_owned
    ConstUnmanaged<impl_scalar_type_2d_view_tpetra> x_remote;
    Unmanaged<impl_scalar_type_2d_view_tpetra> y;

    // AmD information
    const ConstUnmanaged<impl_scalar_type_1d_view> tpetra_values;

    // blocksize
    const local_ordinal_type blocksize_requested;

    // block offsets
    const ConstUnmanaged<i64_3d_view> A_x_offsets;
    const ConstUnmanaged<i64_3d_view> A_x_offsets_remote;

    // diagonal block inverses
    const ConstUnmanaged<btdm_scalar_type_3d_view> d_inv;

    impl_scalar_type damping_factor;

    // When doing a two-stage residual (owned then non-owned), partial_residual is a
    // temporary local multivector that holds the residual (b - Rx) from just the owned entries.
    // It is later combined with the non-owned part during the second stage.
    //
    // This aliases the work buffer impl_->work.
    Unmanaged<impl_scalar_type_2d_view_tpetra> partial_residual;

    ComputeResidualVectorAndSolve(
          const AmD<MatrixType> &amd,
          const BlockTridiags<MatrixType> &btdm,
          const local_ordinal_type &blocksize_requested_,
          const impl_scalar_type damping_factor_,
          const Unmanaged<impl_scalar_type_2d_view_tpetra>& partial_residual_)
      : tpetra_values(amd.tpetra_values),
        blocksize_requested(blocksize_requested_),
        A_x_offsets(amd.A_x_offsets),
        A_x_offsets_remote(amd.A_x_offsets_remote),
        d_inv(btdm.d_inv),
        damping_factor(damping_factor_),
        partial_residual(partial_residual_)
    {}

    template<int B>
    KOKKOS_INLINE_FUNCTION
    void
    operator() (const SinglePassTag<B>&, const member_type &member, magnitude_type& update_norm) const {
      const local_ordinal_type blocksize = (B == 0 ? blocksize_requested : B);
      const local_ordinal_type rowidx = member.league_rank();
      const local_ordinal_type row = rowidx * blocksize;
      const local_ordinal_type num_vectors = b.extent(1);

      impl_scalar_type* xx;
      impl_scalar_type* yy;
      auto A_block_cst = ConstUnmanaged<tpetra_block_access_view_type>(NULL, blocksize, blocksize);

      // Get shared allocation for a local copy of x, Ax, and A
      impl_scalar_type * local_Ax = reinterpret_cast<impl_scalar_type *>(member.team_scratch(0).get_shmem(blocksize*sizeof(impl_scalar_type)));
      impl_scalar_type * local_DinvAx = reinterpret_cast<impl_scalar_type *>(member.team_scratch(0).get_shmem(blocksize*sizeof(impl_scalar_type)));
      impl_scalar_type * local_x = reinterpret_cast<impl_scalar_type *>(member.thread_scratch(0).get_shmem(blocksize*sizeof(impl_scalar_type)));

      for (local_ordinal_type col = 0; col < num_vectors; ++col) {
        if(col)
          member.team_barrier();
        // y -= Rx
        // Initialize accumulation arrays
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, blocksize),[&](const local_ordinal_type & i){
          local_DinvAx[i] = 0;
          local_Ax[i] = b(row + i, col);
        });
        member.team_barrier();

        int numEntries = A_x_offsets.extent(2);

        Kokkos::parallel_for
          (Kokkos::TeamThreadRange(member, 0, numEntries),
          [&](const int k) {
            int64_t A_offset = A_x_offsets(rowidx, 0, k);
            int64_t x_offset = A_x_offsets(rowidx, 1, k);
            if(A_offset != Kokkos::ArithTraits<int64_t>::min()) {
              A_block_cst.assign_data(tpetra_values.data() + A_offset);
              // Pull x into local memory
              size_type remote_cutoff = blocksize * num_local_rows;
              if(x_offset >= remote_cutoff)
                xx = &x_remote(x_offset - remote_cutoff, col);
              else
                xx = &x(x_offset, col);

              Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, blocksize),
                [&](const local_ordinal_type & i){
                  local_x[i] = xx[i];
                });
            
              // matvec on block: local_Ax -= A_block_cst * local_x
              Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(member, blocksize),
                [&](const int k0) {
                  impl_scalar_type val = 0;
                  for(int k1 = 0; k1 < blocksize; k1++)
                    val += A_block_cst(k0, k1) * local_x[k1];
                  Kokkos::atomic_add(local_Ax + k0, -val);
              });
            }
          });
        member.team_barrier();
        // Compute local_DinvAx = D^-1 * local_Ax
        if(member.team_rank() == 0) {
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(member, blocksize),
            [&](const local_ordinal_type &k0) {
              impl_scalar_type val = 0;
              for(int k1=0; k1<blocksize; k1++)
                val += d_inv(rowidx, k0, k1) * local_Ax[k1];
              local_DinvAx[k0] = val;
          });
          // local_DinvAx is fully computed. Now compute the
          // squared y update norm and update y (using damping factor).
          impl_scalar_type norm;
          Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(member, blocksize),
            [&](const local_ordinal_type& k, impl_scalar_type& update) {
              // Compute the change in y (assuming damping_factor == 1) for this entry.
              impl_scalar_type old_y = x(row + k, col);
              impl_scalar_type y_update = local_DinvAx[k] - old_y;
              magnitude_type ydiff = Kokkos::ArithTraits<impl_scalar_type>::abs(y_update);
              update += ydiff * ydiff;
              y(row + k, col) = old_y + damping_factor * y_update;
            }, norm);
          Kokkos::single(Kokkos::PerThread(member),
            [&]() {
              update_norm += norm;
            });
        }
      }
    }

    template<int B>
    KOKKOS_INLINE_FUNCTION
    void
    operator() (const TwoPassOwnedTag<B>&, const member_type &member) const {
      const local_ordinal_type blocksize = (B == 0 ? blocksize_requested : B);
      const local_ordinal_type rowidx = member.league_rank();
      const local_ordinal_type row = rowidx * blocksize;
      const local_ordinal_type num_vectors = b.extent(1);

      auto A_block_cst = ConstUnmanaged<tpetra_block_access_view_type>(NULL, blocksize, blocksize);

      // Get shared allocation for a local copy of x, Ax, and A
      impl_scalar_type * local_Ax = reinterpret_cast<impl_scalar_type *>(member.team_scratch(0).get_shmem(blocksize*sizeof(impl_scalar_type)));
      impl_scalar_type * local_x = reinterpret_cast<impl_scalar_type *>(member.thread_scratch(0).get_shmem(blocksize*sizeof(impl_scalar_type)));

      for (local_ordinal_type col = 0; col < num_vectors; ++col) {
        if(col)
          member.team_barrier();
        // y -= Rx
        // Initialize accumulation arrays
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, blocksize),[&](const local_ordinal_type & i){
          local_Ax[i] = b(row + i, col);
        });
        member.team_barrier();

        int numEntries = A_x_offsets.extent(2);

        Kokkos::parallel_for
          (Kokkos::TeamThreadRange(member, 0, numEntries),
          [&](const int k) {
            int64_t A_offset = A_x_offsets(rowidx, 0, k);
            int64_t x_offset = A_x_offsets(rowidx, 1, k);
            if(A_offset != Kokkos::ArithTraits<int64_t>::min()) {
              A_block_cst.assign_data(tpetra_values.data() + A_offset);
              // Pull x into local memory
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, blocksize),[&](const local_ordinal_type & i){
                local_x[i] = x(x_offset + i, col);
              });
            
              // MatVec op Ax += A*x
              Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(member, blocksize),
                [&](const local_ordinal_type &k0) {
                  impl_scalar_type val = 0;
                  for(int k1=0; k1<blocksize; k1++)
                    val += A_block_cst(k0,k1) * local_x[k1];
                  Kokkos::atomic_add(local_Ax+k0, -val);
              });
            }
          });
        member.team_barrier();
        // Write back the partial residual to yy (temporarily)
        if(member.team_rank() == 0) {
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(member, blocksize),
            [&](const local_ordinal_type &k) {
              y(row + k, col) = local_Ax[k];
          });
        }
      }
    }

    template<int B>
    KOKKOS_INLINE_FUNCTION
    void
    operator() (const TwoPassNonownedTag<B>&, const member_type &member, magnitude_type& update_norm) const {
      const local_ordinal_type blocksize = (B == 0 ? blocksize_requested : B);
      const local_ordinal_type rowidx = member.league_rank();
      const local_ordinal_type row = rowidx * blocksize;
      const local_ordinal_type num_vectors = b.extent(1);

      auto A_block_cst = ConstUnmanaged<tpetra_block_access_view_type>(NULL, blocksize, blocksize);

      // Get shared allocation for a local copy of x, Ax, and A
      impl_scalar_type * local_Ax = reinterpret_cast<impl_scalar_type *>(member.team_scratch(0).get_shmem(blocksize*sizeof(impl_scalar_type)));
      impl_scalar_type * local_DinvAx = reinterpret_cast<impl_scalar_type *>(member.team_scratch(0).get_shmem(blocksize*sizeof(impl_scalar_type)));
      impl_scalar_type * local_x = reinterpret_cast<impl_scalar_type *>(member.thread_scratch(0).get_shmem(blocksize*sizeof(impl_scalar_type)));

      for (local_ordinal_type col = 0; col < num_vectors; ++col) {
        if(col)
          member.team_barrier();
        // y -= Rx
        // Initialize accumulation arrays.
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, blocksize),[&](const local_ordinal_type & i){
          local_DinvAx[i] = 0;
          local_Ax[i] = y(row + i, col);
        });
        member.team_barrier();

        int numEntries = A_x_offsets_remote.extent(2);

        Kokkos::parallel_for
          (Kokkos::TeamThreadRange(member, 0, numEntries),
          [&](const int k) {
            int64_t A_offset = A_x_offsets_remote(rowidx, 0, k);
            int64_t x_offset = A_x_offsets_remote(rowidx, 1, k);
            if(A_offset != Kokkos::ArithTraits<int64_t>::min()) {
              A_block_cst.assign_data(tpetra_values.data() + A_offset);
              // Pull x into local memory
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, blocksize),
                [&](const local_ordinal_type & i){
                  local_x[i] = x_remote(x_offset + i, col);
                });
            
              // matvec on block: local_Ax -= A_block_cst * local_x
              Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(member, blocksize),
                [&](const int k0) {
                  impl_scalar_type val = 0;
                  for(int k1 = 0; k1 < blocksize; k1++)
                    val += A_block_cst(k0, k1) * local_x[k1];
                  Kokkos::atomic_add(local_Ax + k0, -val);
              });
            }
          });
        member.team_barrier();
        // Compute local_DinvAx = D^-1 * local_Ax
        if(member.team_rank() == 0) {
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(member, blocksize),
            [&](const local_ordinal_type &k0) {
              impl_scalar_type val = 0;
              for(int k1=0; k1<blocksize; k1++)
                val += d_inv(rowidx, k0, k1) * local_Ax[k1];
              local_DinvAx[k0] = val;
          });
          // local_DinvAx is fully computed. Now compute the
          // squared y update norm and update y (using damping factor).
          impl_scalar_type norm;
          Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(member, blocksize),
            [&](const local_ordinal_type& k, impl_scalar_type& update) {
              // Compute the change in y (assuming damping_factor == 1) for this entry.
              impl_scalar_type old_y = x(row + k, col);
              impl_scalar_type y_update = local_DinvAx[k] - old_y;
              magnitude_type ydiff = Kokkos::ArithTraits<impl_scalar_type>::abs(y_update);
              update += ydiff * ydiff;
              y(row + k, col) = old_y + damping_factor * y_update;
            }, norm);
          Kokkos::single(Kokkos::PerThread(member),
            [&]() {
              update_norm += norm;
            });
        }
      }
    }

    // Launch SinglePass version (owned + nonowned residual, plus Dinv in a single kernel)
    template<typename MultiVectorLocalViewTypeY,
             typename MultiVectorLocalViewTypeB,
             typename MultiVectorLocalViewTypeX,
             typename MultiVectorLocalViewTypeX_Remote>
    magnitude_type run(
             const MultiVectorLocalViewTypeY &y_,
             const MultiVectorLocalViewTypeB &b_,
             const MultiVectorLocalViewTypeX &x_,
             const MultiVectorLocalViewTypeX_Remote &x_remote_) {
      IFPACK2_BLOCKHELPER_PROFILER_REGION_BEGIN;
      IFPACK2_BLOCKHELPER_TIMER_WITH_FENCE("BlockTriDi::ComputeResidualAndSolve::Run", ComputeResidualAndSolve0, execution_space);

      y = y_; b = b_; x = x_; x_remote = x_remote_;

      const local_ordinal_type blocksize = blocksize_requested;
      const local_ordinal_type nrows = d_inv.extent(0);

      impl_scalar_type norm_sq;
#define BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(B) {                \
        const local_ordinal_type team_size = 8; \
        const local_ordinal_type vector_size = 8; \
        const size_t shmem_team_size = 2 * blocksize*sizeof(btdm_scalar_type); \
        const size_t shmem_thread_size = blocksize*sizeof(btdm_scalar_type); \
        Kokkos::TeamPolicy<execution_space,SinglePassTag<B> >      \
          policy(nrows, team_size, vector_size);    \
        policy.set_scratch_size(0,Kokkos::PerTeam(shmem_team_size),Kokkos::PerThread(shmem_thread_size)); \
        Kokkos::parallel_reduce                                        \
          ("ComputeResidualAndSolve::TeamPolicy::run",            \
           policy, *this, norm_sq); \
      } break
      switch (blocksize_requested) {
        case   3: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 3);
        case   5: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 5);
        case   7: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 7);
        case   9: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 9);
        case  10: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(10);
        case  11: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(11);
        case  16: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(16);
        case  17: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(17);
        case  18: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(18);
        default : BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 0);
      }
#undef BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL
      IFPACK2_BLOCKHELPER_PROFILER_REGION_END;
      IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space)
      return norm_sq;
    }

    template<typename MultiVectorLocalViewTypeY,
             typename MultiVectorLocalViewTypeB,
             typename MultiVectorLocalViewTypeX,
             typename MultiVectorLocalViewTypeX_Remote>
    magnitude_type run(
             const MultiVectorLocalViewTypeB &b_,
             const MultiVectorLocalViewTypeX &x_,
             const MultiVectorLocalViewTypeX_Remote &x_remote_,
             const MultiVectorLocalViewTypeY &y_
             ) {
      IFPACK2_BLOCKHELPER_PROFILER_REGION_BEGIN;
      IFPACK2_BLOCKHELPER_TIMER_WITH_FENCE("BlockTriDi::ComputeResidualAndSolve::Run", ComputeResidualAndSolve0, execution_space);

      b = b_; x = x_; x_remote = x_remote_; y = y_;

      const local_ordinal_type blocksize = blocksize_requested;
      const local_ordinal_type nrows = d_inv.extent(0);

      impl_scalar_type norm_sq;
#define BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(B) {                \
        const local_ordinal_type team_size = 8; \
        const local_ordinal_type vector_size = 8; \
        const size_t shmem_team_size = 2 * blocksize*sizeof(btdm_scalar_type); \
        const size_t shmem_thread_size = blocksize*sizeof(btdm_scalar_type); \
        Kokkos::TeamPolicy<execution_space,SinglePassTag<B> >      \
          policy(nrows, team_size, vector_size);    \
        policy.set_scratch_size(0,Kokkos::PerTeam(shmem_team_size),Kokkos::PerThread(shmem_thread_size)); \
        Kokkos::parallel_reduce                                        \
          ("ComputeResidualAndSolve::TeamPolicy::run",            \
           policy, *this, norm_sq); \
      } break
      switch (blocksize_requested) {
        case   3: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 3);
        case   5: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 5);
        case   7: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 7);
        case   9: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 9);
        case  10: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(10);
        case  11: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(11);
        case  16: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(16);
        case  17: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(17);
        case  18: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(18);
        default : BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 0);
      }
#undef BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL
      IFPACK2_BLOCKHELPER_PROFILER_REGION_END;
      IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space)
      return norm_sq;
    }

    template<typename MultiVectorLocalViewTypeY,
             typename MultiVectorLocalViewTypeB,
             typename MultiVectorLocalViewTypeX,
             typename MultiVectorLocalViewTypeX_Remote>
    magnitude_type run(
             const MultiVectorLocalViewTypeB &b_,
             const MultiVectorLocalViewTypeX &x_,
             const MultiVectorLocalViewTypeX_Remote &x_remote_,
             const MultiVectorLocalViewTypeY &y_
             ) {
      IFPACK2_BLOCKHELPER_PROFILER_REGION_BEGIN;
      IFPACK2_BLOCKHELPER_TIMER_WITH_FENCE("BlockTriDi::ComputeResidualAndSolve::Run", ComputeResidualAndSolve0, execution_space);

      b = b_; x = x_; x_remote = x_remote_; y = y_;

      const local_ordinal_type blocksize = blocksize_requested;
      const local_ordinal_type nrows = d_inv.extent(0);

      impl_scalar_type norm_sq;
#define BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(B) {                \
        const local_ordinal_type team_size = 8; \
        const local_ordinal_type vector_size = 8; \
        const size_t shmem_team_size = 2 * blocksize*sizeof(btdm_scalar_type); \
        const size_t shmem_thread_size = blocksize*sizeof(btdm_scalar_type); \
        Kokkos::TeamPolicy<execution_space,SinglePassTag<B> >      \
          policy(nrows, team_size, vector_size);    \
        policy.set_scratch_size(0,Kokkos::PerTeam(shmem_team_size),Kokkos::PerThread(shmem_thread_size)); \
        Kokkos::parallel_reduce                                        \
          ("ComputeResidualAndSolve::TeamPolicy::run",            \
           policy, *this, norm_sq); \
      } break
      switch (blocksize_requested) {
        case   3: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 3);
        case   5: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 5);
        case   7: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 7);
        case   9: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 9);
        case  10: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(10);
        case  11: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(11);
        case  16: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(16);
        case  17: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(17);
        case  18: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(18);
        default : BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 0);
      }
#undef BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL
      IFPACK2_BLOCKHELPER_PROFILER_REGION_END;
      IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space)
      return norm_sq;
    }

  };

} // namespace Ifpack2::BlockHelperDetails

#endif
