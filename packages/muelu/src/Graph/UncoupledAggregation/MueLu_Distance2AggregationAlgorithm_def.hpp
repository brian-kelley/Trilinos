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
/*
 * MueLu_Distance2AggregationAlgorithm_def.hpp
 *
 *  Created on: Feb 22, 2018
 *      Author: Brian Kelley
 */

#ifndef MUELU_DISTANCE2AGGREGATIONALGORITHM_DEF_HPP_
#define MUELU_DISTANCE2AGGREGATIONALGORITHM_DEF_HPP_

#include "MueLu_GraphBase.hpp"
#include "MueLu_Aggregates.hpp"

#ifdef HAVE_MUELU_KOKKOS_REFACTOR
#include "MueLu_LWGraph_kokkos.hpp"
#include "MueLu_Aggregates_kokkos.hpp"
#endif

#include <vector>

namespace MueLu {
namespace Dist2Impl
{
  //These two versions of getGraph are needed to handle the minor
  //difference between GraphBase and LWGraph_kokkos interfaces
  template<typename LO, typename GO, typename NO>
  static void getGraph(const LWGraph_kokkos<LO, GO, NO>& graph, std::vector<GO>& rowptrs, std::vector<LO>& colinds, LO& maxCol)
  {
    maxCol = -1;
    auto numRows = graph.GetNodeNumVertices(); 
    rowptrs.reserve(numRows + 1);
    colinds.reserve(graph.GetNodeNumEdges());
    for(LO row = 0; row < numRows; row++)
    {
      auto entries = graph.getNeighborVertices(row);
      for(LO i = 0; i < entries.length; i++)
      {
        colinds.push_back(entries(i));
        if(entries(i) > maxCol)
        {
          maxCol = entries(i);
        }
      }
      rowptrs.push_back(colinds.size());
    }
  }

  template<typename LO, typename GO, typename NO>
  static void getGraph(const GraphBase<LO, GO, NO>& graph, std::vector<GO>& rowptrs, std::vector<LO>& colinds, LO& maxCol)
  {
    maxCol = -1;
    auto numRows = graph.GetNodeNumVertices(); 
    rowptrs.reserve(numRows + 1);
    colinds.reserve(graph.GetNodeNumEdges());
    for(LO row = 0; row < numRows; row++)
    {
      auto entries = graph.getNeighborVertices(row);
      for(LO i = 0; i < entries.size(); i++)
      {
        colinds.push_back(entries[i]);
        if(entries[i] > maxCol)
        {
          maxCol = entries[i];
        }
      }
      rowptrs.push_back(colinds.size());
    }
  }
}

template <class AggregatesType, class GraphType, class LocalOrdinal, class GlobalOrdinal, class Node>
void BuildAggregatesDistance2(
    const GraphType& graph, AggregatesType& aggregates,
    std::vector<unsigned>& aggStat, LocalOrdinal& numNonAggregatedNodes, LocalOrdinal maxAggSize)
{
  typedef LocalOrdinal LO;
  const LO  numRows = graph.GetNodeNumVertices();
  const int myRank  = graph.GetComm()->getRank();

  ArrayRCP<LO> vertex2AggId = aggregates.GetVertex2AggId()->getDataNonConst(0);
  ArrayRCP<LO> procWinner   = aggregates.GetProcWinner()  ->getDataNonConst(0);

  LO numLocalAggregates = aggregates.GetNumAggregates();
  //get the sparse local graph in CRS
  std::vector<GlobalOrdinal> rowptrs;
  std::vector<LocalOrdinal> colinds;

  rowptrs.push_back(0);
  //need precise number of columns for SPGEMM
  LO maxCol = -1;
  Dist2Impl::getGraph(graph, rowptrs, colinds, maxCol);

  //the local CRS graph to Kokkos device views, then compute graph squared
  typedef typename Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node>::local_graph_type graph_t;
  typedef typename graph_t::device_type device_t;
  typedef typename device_t::memory_space memory_space;
  typedef typename device_t::execution_space execution_space;
  typedef typename graph_t::row_map_type::non_const_type rowptrs_view;
  typedef Kokkos::View<size_t*, Kokkos::HostSpace> host_rowptrs_view;
  typedef typename graph_t::entries_type::non_const_type colinds_view;
  typedef Kokkos::View<LocalOrdinal*, Kokkos::HostSpace> host_colinds_view;
  //the values are never used, but need this to pass dummy views to spgemm_numeric
  typedef Kokkos::View<double*, memory_space> values_view;
  //note: just using colinds_view in place of scalar_view_t type (it won't be used at all by symbolic SPGEMM)
  typedef KokkosKernels::Experimental::KokkosKernelsHandle<
    typename rowptrs_view::const_value_type, typename colinds_view::const_value_type, typename values_view::const_value_type, 
             execution_space, memory_space, memory_space> KernelHandle;

  KernelHandle kh;
  //leave gc/spgemm algorithm choices as the default
  kh.create_spgemm_handle();
  //kh.create_graph_coloring_handle(KokkosGraph::COLORING_D2_MATRIX_SQUARED);
  kh.create_graph_coloring_handle();

  auto numCols = maxCol + 1;

  //Create device views for graph rowptrs/colinds
  rowptrs_view aRowptrs("A device rowptrs", numRows + 1);
  colinds_view aColinds("A device colinds", colinds.size());

  // Populate A in temporary host views, then copy to device
  {
    host_rowptrs_view aHostRowptrs("A host rowptrs", numRows + 1);
    host_colinds_view aHostColinds("A host colinds", colinds.size());
    for(LO i = 0; i < numRows + 1; i++)
    {
      aHostRowptrs(i) = rowptrs[i];
    }
    for(size_t i = 0; i < colinds.size(); i++)
    {
      aHostColinds(i) = colinds[i];
    }
    Kokkos::deep_copy(aRowptrs, aHostRowptrs);
    Kokkos::deep_copy(aColinds, aHostColinds);
  }

  //Get explicit transpose of A
  //note: this View constructor will 0-initialize aTransRowptrs (needed by transpose_graph)
  // Uncomment this when the KokkosKernels shapshot with Will McLendon's dist2 is added to Trilinos
  rowptrs_view aTransRowptrs("A^T device rowptrs", numCols + 1);
  colinds_view aTransColinds("A^T device colinds", colinds.size());
  KokkosKernels::Impl::transpose_graph
    <rowptrs_view, colinds_view, rowptrs_view, colinds_view, rowptrs_view, execution_space>
    (numRows, numCols, aRowptrs, aColinds, aTransRowptrs, aTransColinds);

  /*
  //run d2 graph coloring on a2
  KokkosGraph::Experimental::graph_color_d2(&kh, numRows, numCols, aRowptrs, aColinds, aTransRowptrs, aTransColinds);
  */
  // For now, use explicit dist2 = dist1(spgemm(A, A^T))
  //get AA^T (symbolic)
  //m = n = k = numRows because all matrices are square
  rowptrs_view a2Rowptrs("AA^T device rowptrs", numRows + 1);
  KokkosSparse::Experimental::spgemm_symbolic(&kh, numRows, numCols, numRows,
      aRowptrs, aColinds, false, aTransRowptrs, aTransColinds, false,
      a2Rowptrs);

  //allocate AA^T colinds
  auto a2nnz = kh.get_spgemm_handle()->get_c_nnz();
  colinds_view a2Colinds("AA^T device colidns", a2nnz);
  //scope a_dummy/a2_dummy so they are destructed right after numeric
  {
    values_view a_dummy("dummy A values", aColinds.dimension_0());
    values_view a2_dummy("dummy AA^T values", a2nnz);
    KokkosSparse::Experimental::spgemm_numeric(&kh, numRows, numCols, numRows,
        aRowptrs, aColinds, a_dummy, false, aTransRowptrs, aTransColinds, a_dummy, false,
        a2Rowptrs, a2Colinds, a2_dummy);
  }

  KokkosGraph::Experimental::graph_color(&kh, numRows, numCols, a2Rowptrs, a2Colinds);

  // extract the colors
  auto coloringHandle = kh.get_graph_coloring_handle();
  auto colorsDevice = coloringHandle->get_vertex_colors();

  auto colors = Kokkos::create_mirror_view(colorsDevice);
  Kokkos::deep_copy(colors, colorsDevice);

  //clean up coloring handle
  kh.destroy_graph_coloring_handle();

  //have color 1 (first color) be the aggregate roots (add those to mapping first)
  LocalOrdinal aggCount = 0;
  for(LocalOrdinal i = 0; i < numRows; i++)
  {
    if(colors(i) == 1 && aggStat[i] == READY)
    {
      vertex2AggId[i] = aggCount++;
      aggStat[i] = AGGREGATED;
      numLocalAggregates++;
      procWinner[i] = myRank;
    }
  }
  numNonAggregatedNodes = 0;
  std::vector<LocalOrdinal> aggSizes(numLocalAggregates, 0);
  for(int i = 0; i < numRows; i++)
  {
    if(vertex2AggId[i] >= 0)
      aggSizes[vertex2AggId[i]]++;
  }
  //now assign every READY vertex to a directly connected root
  for(LocalOrdinal i = 0; i < numRows; i++)
  {
    if(colors(i) != 1 && (aggStat[i] == READY || aggStat[i] == NOTSEL))
    {
      //get neighbors of vertex i and
      //look for local, aggregated, color 1 neighbor (valid root)
      for(size_t j = rowptrs[i]; j < rowptrs[i + 1]; j++)
      {
        auto nei = colinds[j];
        LocalOrdinal agg = vertex2AggId[nei];
        if(graph.isLocalNeighborVertex(nei) && colors(nei) == 1 && aggStat[nei] == AGGREGATED && aggSizes[agg] < maxAggSize)
        {
          //assign vertex i to aggregate with root j
          vertex2AggId[i] = agg;
          aggSizes[agg]++;
          aggStat[i] = AGGREGATED;
          procWinner[i] = myRank;
          break;
        }
      }
    }
    if(aggStat[i] != AGGREGATED)
    {
      numNonAggregatedNodes++;
      if(aggStat[i] == NOTSEL)
        aggStat[i] = READY;
    }
  }
  // update aggregate object
  aggregates.SetNumAggregates(numLocalAggregates);
}

}  //namespace MueLu::

#endif

