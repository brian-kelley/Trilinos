// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
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
#ifndef MUELU_AGGREGATIONPHASE3ALGORITHM_KOKKOS_DEF_HPP
#define MUELU_AGGREGATIONPHASE3ALGORITHM_KOKKOS_DEF_HPP

#ifdef HAVE_MUELU_KOKKOS_REFACTOR

#include <Teuchos_Comm.hpp>
#include <Teuchos_CommHelpers.hpp>

#include <Xpetra_Vector.hpp>

#include "MueLu_AggregationPhase3Algorithm_kokkos_decl.hpp"

#include "MueLu_Aggregates_kokkos.hpp"
#include "MueLu_Exceptions.hpp"
#include "MueLu_LWGraph_kokkos.hpp"
#include "MueLu_Monitor.hpp"

namespace MueLu {

  // Try to stick unaggregated nodes into a neighboring aggregate if they are
  // not already too big. Otherwise, make a new aggregate
  template <class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggregationPhase3Algorithm_kokkos<LocalOrdinal, GlobalOrdinal, Node>::
  BuildAggregates(const ParameterList& params, const LWGraph_kokkos& graph,
                  Aggregates_kokkos& aggregates, Kokkos::View<unsigned*, typename MueLu::
                  LWGraph_kokkos<LO,GO,Node>::local_graph_type::device_type::
                  memory_space>& aggStatView, LO& numNonAggregatedNodes, Kokkos::View<LO*,
                  typename MueLu::LWGraph_kokkos<LO, GO, Node>::local_graph_type::device_type::
                  memory_space>& colorsDevice, LO& numColors) const {
    if(numNonAggregatedNodes == 0)
    {
      //no non-aggregated nodes, so this phase has nothing to do
      return;
    }
    if(params.get<bool>("aggregation: deterministic"))
    {
      BuildAggregatesDeterministic(params, graph, aggregates, aggStatView, numNonAggregatedNodes, colorsDevice, numColors);
      return;
    }
    Monitor m(*this, "BuildAggregates");

    typedef typename MueLu::LWGraph_kokkos<LO, GO, Node>::local_graph_type graph_t;
    typedef typename graph_t::device_type::memory_space memory_space;
    typedef typename graph_t::device_type::execution_space execution_space;

    bool error_on_isolated = false;
    if(params.isParameter("aggregation: error on nodes with no on-rank neighbors"))
      params.get<bool>("aggregation: error on nodes with no on-rank neighbors");

    const LO  numRows = graph.GetNodeNumVertices();
    const int myRank  = graph.GetComm()->getRank();

    auto vertex2AggId = aggregates.GetVertex2AggId()->template getLocalView<memory_space>();
    auto procWinner   = aggregates.GetProcWinner()  ->template getLocalView<memory_space>();

    LO numLocalAggregates = aggregates.GetNumAggregates();

    // Create views for values accumulation
    Kokkos::View<LO, memory_space> numLocalAggregatesView("numLocalAggregates");
    typename Kokkos::View<LO, memory_space>::HostMirror h_numLocalAggregatesView =
      Kokkos::create_mirror_view (numLocalAggregatesView);
    h_numLocalAggregatesView() = numLocalAggregates;
    Kokkos::deep_copy(numLocalAggregatesView, h_numLocalAggregatesView);

    // Create an extra view to keep track of throw flags
    Kokkos::View<bool, memory_space> phase3Flags("phase 3 flags");
    typename Kokkos::View<bool, memory_space>::HostMirror h_phase3Flags =
      Kokkos::create_mirror_view (phase3Flags);
    h_phase3Flags() = false;
    Kokkos::deep_copy(phase3Flags, h_phase3Flags);

    Kokkos::View<LO, memory_space> numNonAggregatedNodesView("numNonAggregatedNodes");
    typename Kokkos::View<LO, memory_space>::HostMirror h_numNonAggregatedNodesView =
      Kokkos::create_mirror_view (numNonAggregatedNodesView);
    h_numNonAggregatedNodesView() = numNonAggregatedNodes;
    Kokkos::deep_copy(numNonAggregatedNodesView, h_numNonAggregatedNodesView);

    for(LO color = 2; color <= numColors; color++)
    {
      Kokkos::parallel_for("Aggregation Phase 3: determining new roots", numRows,
        KOKKOS_LAMBDA (const LO i) {
          if (colorsDevice(i) != color || aggStatView(i) == AGGREGATED || aggStatView(i) == IGNORED) {
            return;
          }
          auto neighOfINode = graph.getNeighborVertices(i);
          // We don't want a singleton. So lets see if there is an unaggregated
          // neighbor that we can also put with this point.
          bool isNewAggregate = false;
          // Guaranteed to create a new aggregate with root i
          for (int j = 0; j < as<int>(neighOfINode.length); j++) {
            LO neigh = neighOfINode(j);
            if (neigh != i && graph.isLocalNeighborVertex(neigh)
                && aggStatView(neigh) == READY) {
              isNewAggregate = true;
              break;
            }
          }

          if (isNewAggregate) {
            LO aggNeighbors = 0;
            LO newAggID = Kokkos::atomic_fetch_add(&numLocalAggregatesView(), 1);
            // Guaranteed to create a new aggregate with root i
            for (int j = 0; j < as<int>(neighOfINode.length); j++) {
              LO neigh = neighOfINode(j);
              if (neigh != i && graph.isLocalNeighborVertex(neigh)
                  && aggStatView(neigh) == READY) {
                aggStatView (neigh)    = AGGREGATED;
                vertex2AggId(neigh, 0) = newAggID;
                procWinner  (neigh, 0) = myRank;
                aggNeighbors++;
              }
            }
            // Create new aggregate (not singleton)
            // aggregates.SetIsRoot(i);
            vertex2AggId(i, 0) = newAggID;
            //aggNeighbors plus the root are newly aggregated
            Kokkos::atomic_fetch_add(&numNonAggregatedNodesView(), -aggNeighbors);
          } else {
            // We do not want a singleton, but there are no non-aggregated
            // neighbors. Lets see if we can connect to any other aggregates
            // NOTE: This is very similar to phase 2b, but simpler: we stop with
            // the first found aggregate
            int j = 0;
            for (; j < as<int>(neighOfINode.length); ++j) {
              LO neigh = neighOfINode(j);

              // We don't check (neigh != rootCandidate), as it is covered by checking (aggStatView(neigh) == AGGREGATED)
              if (graph.isLocalNeighborVertex(neigh)
                  && aggStatView(neigh) == AGGREGATED)
                break;
            }

            if (j < as<int>(neighOfINode.length)) {
              // Assign to an adjacent aggregate
              vertex2AggId(i, 0) = vertex2AggId(neighOfINode(j), 0);
            } else if (error_on_isolated) {
              // Error on this isolated node, as the user has requested
              phase3Flags() = true;
            } else {
              // Create new aggregate (singleton)
              vertex2AggId(i, 0) = Kokkos::atomic_fetch_add(&numLocalAggregatesView(), 1);
            }
          }

          // One way or another, the node is aggregated (possibly into a singleton)
          aggStatView(i, 0) = AGGREGATED;
          procWinner (i, 0) = myRank;
          Kokkos::atomic_fetch_add(&numNonAggregatedNodesView(), -1);
        });
    }
    Kokkos::deep_copy(h_numLocalAggregatesView, numLocalAggregatesView);
    Kokkos::deep_copy(h_numNonAggregatedNodesView, numNonAggregatedNodesView);

    Kokkos::deep_copy(h_phase3Flags, phase3Flags);
    if(h_phase3Flags() == true) {
      std::ostringstream oss;
      oss << "MueLu::AggregationPhase3Algorithm::BuildAggregates: MueLu has detected a non-Dirichlet node that has no on-rank neighbors and is terminating (by user request). "<<std::endl;
      oss << "If this error is being generated at level 0, this is due to an initial partitioning problem in your matrix."<<std::endl;
      oss << "If this error is being generated at any other level, try turning on repartitioning, which may fix this problem."<<std::endl;
      throw Exceptions::RuntimeError(oss.str());
    }

    numLocalAggregates    = h_numLocalAggregatesView();
    numNonAggregatedNodes = h_numNonAggregatedNodesView();

    numLocalAggregates    = h_numLocalAggregatesView();
    numNonAggregatedNodes = h_numNonAggregatedNodesView();

    // update aggregate object
    aggregates.SetNumAggregates(numLocalAggregates);
  }

  // Try to stick unaggregated nodes into a neighboring aggregate if they are
  // not already too big. Otherwise, make a new aggregate
  template <class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggregationPhase3Algorithm_kokkos<LocalOrdinal, GlobalOrdinal, Node>::
  BuildAggregatesDeterministic(const ParameterList& params, const LWGraph_kokkos& graph,
                  Aggregates_kokkos& aggregates, Kokkos::View<unsigned*, typename MueLu::
                  LWGraph_kokkos<LO,GO,Node>::local_graph_type::device_type::
                  memory_space>& aggStatView, LO& numNonAggregatedNodes, Kokkos::View<LO*,
                  typename MueLu::LWGraph_kokkos<LO, GO, Node>::local_graph_type::device_type::
                  memory_space>& colorsDevice, LO& numColors) const {
    Monitor m(*this, "BuildAggregates (deterministic)");

    typedef typename MueLu::LWGraph_kokkos<LO, GO, Node>::local_graph_type graph_t;
    typedef typename graph_t::device_type::memory_space memory_space;
    typedef typename graph_t::device_type::execution_space execution_space;

    bool error_on_isolated = false;
    if(params.isParameter("aggregation: error on nodes with no on-rank neighbors"))
      params.get<bool>("aggregation: error on nodes with no on-rank neighbors");

    const LO  numRows = graph.GetNodeNumVertices();
    const int myRank  = graph.GetComm()->getRank();

    auto vertex2AggId = aggregates.GetVertex2AggId()->template getLocalView<memory_space>();
    auto procWinner   = aggregates.GetProcWinner()  ->template getLocalView<memory_space>();

    LO numLocalAggregates = aggregates.GetNumAggregates();

    // Create views for values accumulation
    Kokkos::View<LO, memory_space> numLocalAggregatesView("numLocalAggregates");
    typename Kokkos::View<LO, memory_space>::HostMirror h_numLocalAggregatesView =
      Kokkos::create_mirror_view (numLocalAggregatesView);
    h_numLocalAggregatesView() = numLocalAggregates;
    Kokkos::deep_copy(numLocalAggregatesView, h_numLocalAggregatesView);

    // Create an extra view to keep track of throw flags
    Kokkos::View<bool, memory_space> phase3Flags("phase 3 flags");
    typename Kokkos::View<bool, memory_space>::HostMirror h_phase3Flags =
      Kokkos::create_mirror_view (phase3Flags);
    h_phase3Flags() = false;
    Kokkos::deep_copy(phase3Flags, h_phase3Flags);

    Kokkos::View<LO, memory_space> numNonAggregatedNodesView("numNonAggregatedNodes");
    typename Kokkos::View<LO, memory_space>::HostMirror h_numNonAggregatedNodesView =
      Kokkos::create_mirror_view (numNonAggregatedNodesView);
    h_numNonAggregatedNodesView() = numNonAggregatedNodes;
    Kokkos::deep_copy(numNonAggregatedNodesView, h_numNonAggregatedNodesView);

    Kokkos::View<LO*, memory_space> newRoots("New root LIDs", numNonAggregatedNodes);
    Kokkos::View<LO, memory_space> numNewRoots("Number of new aggregates of current color");
    auto h_numNewRoots = Kokkos::create_mirror_view(numNewRoots);

    for(LO color = 1; color <= numColors; color++)
    {
      h_numNewRoots() = 0;
      Kokkos::deep_copy(numNewRoots, h_numNewRoots);
      //first parallel_for determine set of new root nodes (including singletons), AND
      //joins nodes to adjacent aggregates where possible
      Kokkos::parallel_for("Aggregation Phase 3: determining new roots", numRows,
      KOKKOS_LAMBDA (const LO i) {
        if (colorsDevice(i) != color || aggStatView(i) == AGGREGATED || aggStatView(i) == IGNORED) {
          return;
        }
        auto neighOfINode = graph.getNeighborVertices(i);
        // We don't want a singleton. So lets see if there is an unaggregated
        // neighbor that we can also put with this point.
        bool isNewAggregate = false;
        // Guaranteed to create a new aggregate with root i
        for (int j = 0; j < as<int>(neighOfINode.length); j++) {
          LO neigh = neighOfINode(j);
          if (neigh != i && graph.isLocalNeighborVertex(neigh)
              && aggStatView(neigh) == READY) {
            isNewAggregate = true;
            break;
          }
        }

        if (isNewAggregate) {
          newRoots(Kokkos::atomic_fetch_add(&numNewRoots(), 1)) = i;
        } else {
          // We do not want a singleton, but there are no non-aggregated
          // neighbors. Lets see if we can connect to any other aggregates
          // NOTE: This is very similar to phase 2b, but simpler: we stop with
          // the first found aggregate
          int j = 0;
          for (; j < as<int>(neighOfINode.length); ++j) {
            LO neigh = neighOfINode(j);

            // We don't check (neigh != rootCandidate), as it is covered by checking (aggStatView(neigh) == AGGREGATED)
            if (graph.isLocalNeighborVertex(neigh)
                && aggStatView(neigh) == AGGREGATED)
              break;
          }

          if (j < as<int>(neighOfINode.length)) {
            // Assign to an adjacent aggregate
            vertex2AggId(i, 0) = vertex2AggId(neighOfINode(j), 0);
            Kokkos::atomic_fetch_add(&numNonAggregatedNodesView(), -1);
          } else if (error_on_isolated) {
            // Error on this isolated node, as the user has requested
            phase3Flags() = true;
          } else {
            // Will create new aggregate (singleton)
            newRoots(Kokkos::atomic_fetch_add(&numNewRoots(), 1)) = i;
          }
        }
      });

      Kokkos::deep_copy(h_phase3Flags, phase3Flags);
      if(h_phase3Flags() == true) {
        std::ostringstream oss;
        oss << "MueLu::AggregationPhase3Algorithm::BuildAggregates: MueLu has detected a non-Dirichlet node that has no on-rank neighbors and is terminating (by user request). "<<std::endl;
        oss << "If this error is being generated at level 0, this is due to an initial partitioning problem in your matrix."<<std::endl;
        oss << "If this error is being generated at any other level, try turning on repartitioning, which may fix this problem."<<std::endl;
        throw Exceptions::RuntimeError(oss.str());
      }

      //if new aggregates are being created, sort the set of roots so that
      //aggregate IDs are deterministic
      Kokkos::deep_copy(h_numNewRoots, numNewRoots);
      std::cout << "Phase 3: creating " << h_numNewRoots() << " new aggregates with root color " << color << '\n';
      if(h_numNewRoots() > 0)
      {
        Kokkos::sort(newRoots, 0, h_numNewRoots());
        Kokkos::parallel_for("Aggregation Phase 3: determining new roots", h_numNewRoots(),
        KOKKOS_LAMBDA (const LO newRootIndex) {
          LO i = newRoots(newRootIndex);
          LO newAggID = numLocalAggregatesView() + newRootIndex;
          auto neighOfINode = graph.getNeighborVertices(i);
          // Guaranteed to create a new aggregate with root i
          // and its unaggregated neighbors (if any)

          LO aggSize = 1;
          // Guaranteed to create a new aggregate with root i
          for (int j = 0; j < as<int>(neighOfINode.length); j++) {
            LO neigh = neighOfINode(j);
            if (neigh != i && graph.isLocalNeighborVertex(neigh)
                && aggStatView(neigh) == READY) {
              aggStatView (neigh)    = AGGREGATED;
              vertex2AggId(neigh, 0) = newAggID;
              procWinner  (neigh, 0) = myRank;
              aggSize++;
            }
          }
          aggStatView (i)    = AGGREGATED;
          vertex2AggId(i, 0) = newAggID;
          procWinner  (i, 0) = myRank;
          Kokkos::atomic_fetch_add(&numNonAggregatedNodesView(), -aggSize);
        });
      }
    }
    Kokkos::deep_copy(h_numLocalAggregatesView, numLocalAggregatesView);
    Kokkos::deep_copy(h_numNonAggregatedNodesView, numNonAggregatedNodesView);

    numLocalAggregates    = h_numLocalAggregatesView();
    numNonAggregatedNodes = h_numNonAggregatedNodesView();

    // update aggregate object
    aggregates.SetNumAggregates(numLocalAggregates);
  }

} // end namespace

#endif // HAVE_MUELU_KOKKOS_REFACTOR
#endif // MUELU_AGGREGATIONPHASE3ALGORITHM_KOKKOS_DEF_HPP
