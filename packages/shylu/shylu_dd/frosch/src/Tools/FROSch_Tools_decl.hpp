//@HEADER
// ************************************************************************
//
//               ShyLU: Hybrid preconditioner package
//                 Copyright 2012 Sandia Corporation
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
// Questions? Contact Alexander Heinlein (alexander.heinlein@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#ifndef _FROSCH_TOOLS_DECL_HPP
#define _FROSCH_TOOLS_DECL_HPP

#ifndef FROSCH_ASSERT
#define FROSCH_ASSERT(A,S) TEUCHOS_TEST_FOR_EXCEPTION(!(A),std::logic_error,S);
#endif

#ifndef FROSCH_TIMER_START
#define FROSCH_TIMER_START(A,S) RCP<TimeMonitor> A = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(std::string("FROSch: ") + std::string(S))));
#endif

#ifndef FROSCH_TIMER_START_LEVELID
#define FROSCH_TIMER_START_LEVELID(A,S) RCP<TimeMonitor> A = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(std::string("FROSch: ") + std::string(S) + " (Level " + std::to_string(this->LevelID_) + std::string(")"))));
#endif

#ifndef FROSCH_TIMER_STOP
#define FROSCH_TIMER_STOP(A) A.reset();
#endif

#ifndef FROSCH_WARNING
#define FROSCH_WARNING(CLASS,VERBOSE,OUTPUT) if (VERBOSE) std::cerr << CLASS << " : WARNING: " << OUTPUT << std::endl;
#endif

#ifndef FROSCH_NOTIFICATION
#define FROSCH_NOTIFICATION(CLASS,VERBOSE,OUTPUT) if (VERBOSE) std::cout << CLASS << " : NOTIFICATION: " << OUTPUT << std::endl;
#endif

#ifndef FROSCH_TEST_OUTPUT
#define FROSCH_TEST_OUTPUT(COMM,VERBOSE,OUTPUT) COMM->barrier(); COMM->barrier(); COMM->barrier(); if (VERBOSE) std::cout << OUTPUT << std::endl;
#endif

#include <ShyLU_DDFROSch_config.h>

#include <Tpetra_Distributor.hpp>

#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_CrsGraphFactory.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_VectorFactory.hpp>
#include <Xpetra_ExportFactory.hpp>

#ifdef HAVE_SHYLU_DDFROSCH_ZOLTAN2
#include <Zoltan2_MatrixAdapter.hpp>
#include <Zoltan2_XpetraCrsMatrixAdapter.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#endif


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    #if defined HAVE_XPETRA_EPETRA || defined HAVE_TPETRA_INT_INT
    typedef int DefaultGlobalOrdinal;
    #elif !defined HAVE_TPETRA_INT_LONG_LONG
    typedef long DefaultGlobalOrdinal;
    #else
    typedef long long DefaultGlobalOrdinal;
    #endif

    enum DofOrdering {NodeWise=0,DimensionWise=1,Custom=2};

    enum NullSpace {LaplaceNullSpace=0,LinearElasticityNullSpace=1};

        enum Verbosity {None=0,All=1};

    template <typename LO,
              typename GO>
    class OverlappingData {

    protected:

        using IntVec        = Array<int>;

        using LOVec         = Array<LO>;

    public:

        OverlappingData(GO gid,
                        int pid,
                        LO lid);

        int Merge(const RCP<OverlappingData<LO,GO> > od) const;

        
        GO GID_;

        mutable IntVec PIDs_;

        mutable LOVec LIDs_;

    };

    template <typename LO,typename GO>
    int MergeList(Array<RCP<OverlappingData<LO,GO> > > &odList);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <typename LO,
              typename GO,
              typename NO>
#else
    template <typename NO>
#endif
    class LowerPIDTieBreak : public Tpetra::Details::TieBreak<LO,GO> {

    protected:

#ifndef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using LO = typename Tpetra::Map<>::local_ordinal_type;
        using GO = typename Tpetra::Map<>::global_ordinal_type;
#endif
        using CommPtr                   = RCP<const Comm<int> >;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using ConstXMapPtr              = RCP<const Map<LO,GO,NO> >;
#else
        using ConstXMapPtr              = RCP<const Map<NO> >;
#endif

        using OverlappingDataPtr        = RCP<OverlappingData<LO,GO> >;
        using OverlappingDataPtrVec     = Array<OverlappingDataPtr>;

        using UN                        = unsigned;

        using IntVec                    = Array<int>;
        using IntVecVecPtr              = ArrayRCP<IntVec>;

        using LOVec                     = Array<LO>;

        using GOVec                     = Array<GO>;
        using GOVecPtr                  = ArrayRCP<GO>;
        using GOVecVec                  = Array<GOVec>;
        using GOVecVecPtr               = ArrayRCP<GOVec>;

    public:
        LowerPIDTieBreak(CommPtr comm,
                         ConstXMapPtr originalMap,
                         UN dimension,
                         UN levelID = 1); // This is in order to estimate the length of SendImageIDs_ and ExportEntries_ in advance

        virtual bool mayHaveSideEffects() const {
            return false;
        }

        IntVecVecPtr& getComponents()
        {
            return ComponentsSubdomains_;
        }

        int sendDataToOriginalMap();

        virtual std::size_t selectedIndex(GO GID,
                                          const std::vector<std::pair<int,LO> > & pid_and_lid) const;

    protected:

        CommPtr MpiComm_;

        ConstXMapPtr OriginalMap_;

        mutable LO ElementCounter_; // This is mutable such that it can be modified in selectedIndex()

        mutable OverlappingDataPtrVec OverlappingDataList_; // This is mutable such that it can be modified in selectedIndex()

        IntVecVecPtr ComponentsSubdomains_; // This is mutable such that it can be modified in selectedIndex()

        UN LevelID_ = 1;
    };

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class LO,class GO,class NO>
    RCP<const Map<LO,GO,NO> > BuildUniqueMap(const RCP<const Map<LO,GO,NO> > map,
#else
    template <class NO>
    RCP<const Map<NO> > BuildUniqueMap(const RCP<const Map<NO> > map,
#endif
                                             bool useCreateOneToOneMap = true,
                                             RCP<Tpetra::Details::TieBreak<LO,GO> > tieBreak = null);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    ArrayRCP<RCP<const Map<LO,GO,NO> > > BuildRepeatedSubMaps(RCP<const Matrix<SC,LO,GO,NO> > matrix,
                                                              ArrayRCP<const RCP<Map<LO,GO,NO> > > subMaps);

    template <class SC,class LO,class GO,class NO>
    ArrayRCP<RCP<const Map<LO,GO,NO> > > BuildRepeatedSubMaps(RCP<const CrsGraph<LO,GO,NO> > graph,
                                                              ArrayRCP<const RCP<Map<LO,GO,NO> > > subMaps);

    template <class SC,class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > BuildRepeatedMapNonConstOld(RCP<const Matrix<SC,LO,GO,NO> > matrix);

    template <class SC,class LO,class GO,class NO>
    RCP<const Map<LO,GO,NO> > BuildRepeatedMapOld(RCP<const Matrix<SC,LO,GO,NO> > matrix);

    template <class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > BuildRepeatedMapNonConstOld(RCP<const CrsGraph<LO,GO,NO> > graph);

    template <class LO,class GO,class NO>
    RCP<const Map<LO,GO,NO> > BuildRepeatedMapOld(RCP<const CrsGraph<LO,GO,NO> > graph);

    template <class SC,class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > BuildRepeatedMapNonConst(RCP<const Matrix<SC,LO,GO,NO> > matrix);

    template <class SC,class LO,class GO,class NO>
    RCP<const Map<LO,GO,NO> > BuildRepeatedMap(RCP<const Matrix<SC,LO,GO,NO> > matrix);

    template <class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > BuildRepeatedMapNonConst(RCP<const CrsGraph<LO,GO,NO> > graph);

    template <class LO,class GO,class NO>
    RCP<const Map<LO,GO,NO> > BuildRepeatedMap(RCP<const CrsGraph<LO,GO,NO> > graph);

    template <class SC,class LO,class GO,class NO>
    int ExtendOverlapByOneLayer_Old(RCP<const Matrix<SC,LO,GO,NO> > inputMatrix,
                                    RCP<const Map<LO,GO,NO> > inputMap,
                                    RCP<const Matrix<SC,LO,GO,NO> > &outputMatrix,
                                    RCP<const Map<LO,GO,NO> > &outputMap);

    template <class SC,class LO,class GO,class NO>
    int ExtendOverlapByOneLayer(RCP<const Matrix<SC,LO,GO,NO> > inputMatrix,
                                RCP<const Map<LO,GO,NO> > inputMap,
                                RCP<const Matrix<SC,LO,GO,NO> > &outputMatrix,
                                RCP<const Map<LO,GO,NO> > &outputMap);

    template <class LO,class GO,class NO>
    int ExtendOverlapByOneLayer(RCP<const CrsGraph<LO,GO,NO> > inputGraph,
                                RCP<const Map<LO,GO,NO> > inputMap,
                                RCP<const CrsGraph<LO,GO,NO> > &outputGraph,
                                RCP<const Map<LO,GO,NO> > &outputMap);
#else
    template <class SC,class NO>
    ArrayRCP<RCP<const Map<NO> > > BuildRepeatedSubMaps(RCP<const Matrix<SC,NO> > matrix,
                                                              ArrayRCP<const RCP<Map<NO> > > subMaps);

    template <class SC,class NO>
    ArrayRCP<RCP<const Map<NO> > > BuildRepeatedSubMaps(RCP<const CrsGraph<NO> > graph,
                                                              ArrayRCP<const RCP<Map<NO> > > subMaps);

    template <class SC,class NO>
    RCP<Map<NO> > BuildRepeatedMapNonConstOld(RCP<const Matrix<SC,NO> > matrix);

    template <class SC,class NO>
    RCP<const Map<NO> > BuildRepeatedMapOld(RCP<const Matrix<SC,NO> > matrix);

    template <class NO>
    RCP<Map<NO> > BuildRepeatedMapNonConstOld(RCP<const CrsGraph<NO> > graph);

    template <class NO>
    RCP<const Map<NO> > BuildRepeatedMapOld(RCP<const CrsGraph<NO> > graph);

    template <class SC,class NO>
    RCP<Map<NO> > BuildRepeatedMapNonConst(RCP<const Matrix<SC,NO> > matrix);

    template <class SC,class NO>
    RCP<const Map<NO> > BuildRepeatedMap(RCP<const Matrix<SC,NO> > matrix);

    template <class NO>
    RCP<Map<NO> > BuildRepeatedMapNonConst(RCP<const CrsGraph<NO> > graph);

    template <class NO>
    RCP<const Map<NO> > BuildRepeatedMap(RCP<const CrsGraph<NO> > graph);

    template <class SC,class NO>
    int ExtendOverlapByOneLayer_Old(RCP<const Matrix<SC,NO> > inputMatrix,
                                    RCP<const Map<NO> > inputMap,
                                    RCP<const Matrix<SC,NO> > &outputMatrix,
                                    RCP<const Map<NO> > &outputMap);

    template <class SC,class NO>
    int ExtendOverlapByOneLayer(RCP<const Matrix<SC,NO> > inputMatrix,
                                RCP<const Map<NO> > inputMap,
                                RCP<const Matrix<SC,NO> > &outputMatrix,
                                RCP<const Map<NO> > &outputMap);

    template <class NO>
    int ExtendOverlapByOneLayer(RCP<const CrsGraph<NO> > inputGraph,
                                RCP<const Map<NO> > inputMap,
                                RCP<const CrsGraph<NO> > &outputGraph,
                                RCP<const Map<NO> > &outputMap);
#endif
    
    /*! \brief Sort the Xpetra::Map by the global IDs \c x
     * \param[in] inputMap Unsorted input map
     */
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class LO,class GO,class NO>
    RCP<const Map<LO,GO,NO> > SortMapByGlobalIndex(RCP<const Map<LO,GO,NO> > inputMap);
#else
    template <class NO>
    RCP<const Map<NO> > SortMapByGlobalIndex(RCP<const Map<NO> > inputMap);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > AssembleMaps(ArrayView<RCP<const Map<LO,GO,NO> > > mapVector,
#else
    template <class NO>
    RCP<Map<NO> > AssembleMaps(ArrayView<RCP<const Map<NO> > > mapVector,
#endif
                                     ArrayRCP<ArrayRCP<LO> > &partMappings);
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > AssembleSubdomainMap(unsigned numberOfBlocks,
                                             ArrayRCP<ArrayRCP<RCP<const Map<LO,GO,NO> > > > dofsMaps,
#else
    template <class NO>
    RCP<Map<NO> > AssembleSubdomainMap(unsigned numberOfBlocks,
                                             ArrayRCP<ArrayRCP<RCP<const Map<NO> > > > dofsMaps,
#endif
                                             ArrayRCP<unsigned> dofsPerNode);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > MergeMapsNonConst(ArrayRCP<RCP<const Map<LO,GO,NO> > > mapVector);
#else
    template <class NO>
    RCP<Map<NO> > MergeMapsNonConst(ArrayRCP<RCP<const Map<NO> > > mapVector);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class LO,class GO,class NO>
    RCP<const Map<LO,GO,NO> > MergeMaps(ArrayRCP<RCP<const Map<LO,GO,NO> > > mapVector);
#else
    template <class NO>
    RCP<const Map<NO> > MergeMaps(ArrayRCP<RCP<const Map<NO> > > mapVector);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class LO,class GO,class NO>
    int BuildDofMaps(const RCP<const Map<LO,GO,NO> > map,
#else
    template <class NO>
    int BuildDofMaps(const RCP<const Map<NO> > map,
#endif
                     unsigned dofsPerNode,
                     unsigned dofOrdering,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                     RCP<const Map<LO,GO,NO> > &nodesMap,
                     ArrayRCP<RCP<const Map<LO,GO,NO> > > &dofMaps,
#else
                     RCP<const Map<NO> > &nodesMap,
                     ArrayRCP<RCP<const Map<NO> > > &dofMaps,
#endif
                     GO offset = 0);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class LO,class GO,class NO>
    int BuildDofMapsVec(const ArrayRCP<RCP<const Map<LO,GO,NO> > > mapVec,
#else
    template <class NO>
    int BuildDofMapsVec(const ArrayRCP<RCP<const Map<NO> > > mapVec,
#endif
                        ArrayRCP<unsigned> dofsPerNodeVec,
                        ArrayRCP<FROSch::DofOrdering> dofOrderingVec,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                        ArrayRCP<RCP<const Map<LO,GO,NO> > > &nodesMapVec,
                        ArrayRCP<ArrayRCP<RCP<const Map<LO,GO,NO> > > >&dofMapsVec);
#else
                        ArrayRCP<RCP<const Map<NO> > > &nodesMapVec,
                        ArrayRCP<ArrayRCP<RCP<const Map<NO> > > >&dofMapsVec);
#endif


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > BuildMapFromDofMaps(const ArrayRCP<RCP<Map<LO,GO,NO> > > &dofMaps,
#else
    template <class NO>
    RCP<Map<NO> > BuildMapFromDofMaps(const ArrayRCP<RCP<Map<NO> > > &dofMaps,
#endif
                                            unsigned dofsPerNode,
                                            unsigned dofOrdering);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class LO,class GO,class NO>
    RCP<Map<LO,GO,NO> > BuildMapFromNodeMap(RCP<Map<LO,GO,NO> > &nodesMap,
#else
    template <class NO>
    RCP<Map<NO> > BuildMapFromNodeMap(RCP<Map<NO> > &nodesMap,
#endif
                                            unsigned dofsPerNode,
                                            unsigned dofOrdering);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class LO,class GO,class NO>
    ArrayRCP<RCP<const Map<LO,GO,NO> > > BuildNodeMapsFromDofMaps(ArrayRCP<ArrayRCP<RCP<const Map<LO,GO,NO> > > >dofsMapsVecVec,
#else
    template <class NO>
    ArrayRCP<RCP<const Map<NO> > > BuildNodeMapsFromDofMaps(ArrayRCP<ArrayRCP<RCP<const Map<NO> > > >dofsMapsVecVec,
#endif
                                                            ArrayRCP<unsigned> dofsPerNodeVec,
                                                            ArrayRCP<DofOrdering> dofOrderingVec);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class LO,class GO,class NO>
    ArrayRCP<RCP<Map<LO,GO,NO> > > BuildSubMaps(RCP<const Map<LO,GO,NO> > &fullMap,
#else
    template <class NO>
    ArrayRCP<RCP<Map<NO> > > BuildSubMaps(RCP<const Map<NO> > &fullMap,
#endif
                                                ArrayRCP<GO> maxSubGIDVec);
    
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    ArrayRCP<GO> FindOneEntryOnlyRowsGlobal(RCP<const Matrix<SC,LO,GO,NO> > matrix,
                                            RCP<const Map<LO,GO,NO> > repeatedMap);

    template <class LO,class GO,class NO>
    ArrayRCP<GO> FindOneEntryOnlyRowsGlobal(RCP<const CrsGraph<LO,GO,NO> > graph,
                                            RCP<const Map<LO,GO,NO> > repeatedMap);
#else
    template <class SC,class NO>
    ArrayRCP<GO> FindOneEntryOnlyRowsGlobal(RCP<const Matrix<SC,NO> > matrix,
                                            RCP<const Map<NO> > repeatedMap);

    template <class NO>
    ArrayRCP<GO> FindOneEntryOnlyRowsGlobal(RCP<const CrsGraph<NO> > graph,
                                            RCP<const Map<NO> > repeatedMap);
#endif
    
    template <class SC,class LO>
    bool ismultiple(ArrayView<SC> A,
                    ArrayView<SC> B);

    template<class T>
    inline void sort(T &v);
    
    template<class T>
    inline void sortunique(T &v);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO,class GO,class NO>
    RCP<MultiVector<SC,LO,GO,NO> > ModifiedGramSchmidt(RCP<const MultiVector<SC,LO,GO,NO> > multiVector,
#else
    template <class SC,class NO>
    RCP<MultiVector<SC,NO> > ModifiedGramSchmidt(RCP<const MultiVector<SC,NO> > multiVector,
#endif
                                                       ArrayView<unsigned> zero = ArrayView<unsigned>());

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC, class LO,class GO,class NO>
    RCP<const MultiVector<SC,LO,GO,NO> > BuildNullSpace(unsigned dimension,
#else
    template <class SC,class NO>
    RCP<const MultiVector<SC,NO> > BuildNullSpace(unsigned dimension,
#endif
                                                        unsigned nullSpaceType,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                                                        RCP<const Map<LO,GO,NO> > repeatedMap,
#else
                                                        RCP<const Map<NO> > repeatedMap,
#endif
                                                        unsigned dofsPerNode,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                                                        ArrayRCP<RCP<const Map<LO,GO,NO> > > dofsMaps,
                                                        RCP<const MultiVector<SC,LO,GO,NO> > nodeList = null);
#else
                                                        ArrayRCP<RCP<const Map<NO> > > dofsMaps,
                                                        RCP<const MultiVector<SC,NO> > nodeList = null);
#endif

#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
#else
    template <class SC,class NO>
#endif
    struct ConvertToXpetra {

    public:

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        static RCP<Map<LO,GO,NO> > ConvertMap(UnderlyingLib lib,
#else
        using LO = typename Tpetra::Map<>::local_ordinal_type;
        using GO = typename Tpetra::Map<>::global_ordinal_type;
        static RCP<Map<NO> > ConvertMap(UnderlyingLib lib,
#endif
                                              const Epetra_BlockMap &map,
                                              RCP<const Comm<int> > comm);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        static RCP<Matrix<SC,LO,GO,NO> > ConvertMatrix(UnderlyingLib lib,
#else
        static RCP<Matrix<SC,NO> > ConvertMatrix(UnderlyingLib lib,
#endif
                                                       Epetra_CrsMatrix &matrix,
                                                       RCP<const Comm<int> > comm);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        static RCP<MultiVector<SC,LO,GO,NO> > ConvertMultiVector(UnderlyingLib lib,
#else
        static RCP<MultiVector<SC,NO> > ConvertMultiVector(UnderlyingLib lib,
#endif
                                                                 Epetra_MultiVector &vector,
                                                                 RCP<const Comm<int> > comm);
    };

    template <class SC,class LO,class NO>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    struct ConvertToXpetra<SC,LO,int,NO> {
#else
    struct ConvertToXpetra<SC,NO> {
#endif

    public:

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        static RCP<Map<LO,int,NO> > ConvertMap(UnderlyingLib lib,
#else
        static RCP<Map<NO> > ConvertMap(UnderlyingLib lib,
#endif
                                               const Epetra_BlockMap &map,
                                               RCP<const Comm<int> > comm);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        static RCP<Matrix<SC,LO,int,NO> > ConvertMatrix(UnderlyingLib lib,
#else
        static RCP<Matrix<SC,NO> > ConvertMatrix(UnderlyingLib lib,
#endif
                                                        Epetra_CrsMatrix &matrix,
                                                        RCP<const Comm<int> > comm);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        static RCP<MultiVector<SC,LO,int,NO> > ConvertMultiVector(UnderlyingLib lib,
#else
        static RCP<MultiVector<SC,NO> > ConvertMultiVector(UnderlyingLib lib,
#endif
                                                                  Epetra_MultiVector &vector,
                                                                  RCP<const Comm<int> > comm);
    };

    template <class SC,class LO,class NO>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    struct ConvertToXpetra<SC,LO,long long,NO> {
#else
    struct ConvertToXpetra<SC,NO> {
#endif

    public:

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        static RCP<Map<LO,long long,NO> > ConvertMap(UnderlyingLib lib,
#else
        static RCP<Map<NO> > ConvertMap(UnderlyingLib lib,
#endif
                                                     const Epetra_BlockMap &map,
                                                     RCP<const Comm<int> > comm);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        static RCP<Matrix<SC,LO,long long,NO> > ConvertMatrix(UnderlyingLib lib,
#else
        static RCP<Matrix<SC,NO> > ConvertMatrix(UnderlyingLib lib,
#endif
                                                              Epetra_CrsMatrix &matrix,
                                                              RCP<const Comm<int> > comm);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        static RCP<MultiVector<SC,LO,long long,NO> > ConvertMultiVector(UnderlyingLib lib,
#else
        static RCP<MultiVector<SC,NO> > ConvertMultiVector(UnderlyingLib lib,
#endif
                                                                        Epetra_MultiVector &vector,
                                                                        RCP<const Comm<int> > comm);
    };
#endif

    template <class Type>
    RCP<Type> ExtractPtrFromParameterList(ParameterList& paramList,
                                          std::string namePtr="Ptr");

    template <class Type>
    ArrayRCP<Type> ExtractVectorFromParameterList(ParameterList& paramList,
                                                  std::string nameVector="Vector");

#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class LO,class GO,class NO>
    RCP<Epetra_Map> ConvertToEpetra(const Map<LO,GO,NO> &map,
#else
    template <class NO>
    RCP<Epetra_Map> ConvertToEpetra(const Map<NO> &map,
#endif
                                    RCP<Epetra_Comm> epetraComm);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    RCP<Epetra_MultiVector> ConvertToEpetra(const MultiVector<SC,LO,GO,NO> &vector,
#else
    template <class SC,class NO>
    RCP<Epetra_MultiVector> ConvertToEpetra(const MultiVector<SC,NO> &vector,
#endif
                                            RCP<Epetra_Comm> epetraComm);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    RCP<Epetra_CrsMatrix> ConvertToEpetra(const Matrix<SC,LO,GO,NO> &matrix,
#else
    template <class SC,class NO>
    RCP<Epetra_CrsMatrix> ConvertToEpetra(const Matrix<SC,NO> &matrix,
#endif
                                          RCP<Epetra_Comm> epetraComm);
#endif

    template <class LO>
    Array<LO> GetIndicesFromString(std::string string);

#ifdef HAVE_SHYLU_DDFROSCH_ZOLTAN2
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int RepartionMatrixZoltan2(RCP<Matrix<SC,LO,GO,NO> > &crsMatrix,
#else
    template <class SC,class NO>
    int RepartionMatrixZoltan2(RCP<Matrix<SC,NO> > &crsMatrix,
#endif
                               RCP<ParameterList> parameterList);
#endif

    /*!
    \brief Throw runtime error due to missing package in build configuration

    As many packages are optional, we might detect only at runtime that are certain package
    is not included into the build configuration, but still is used by FROSch.
    Use this routine to throw a generic error message with some information for the user
    and provide details how to fix it.

    \param[in] forschObj FROSch object that is asking for the missing package
    \param[in] packageName Name of the missing package
    */
    inline void ThrowErrorMissingPackage(const std::string& froschObj,
                                         const std::string& packageName)
    {
        // Create the error message
        std::stringstream errMsg;
        errMsg << froschObj << " is asking for the Trilinos packate '"<< packageName << "', "
        "but this package is not included in your build configuration. "
        "Please enable '" << packageName << "' in your build configuration to be used with ShyLU_DDFROSch.";

        // Throw the error
        FROSCH_ASSERT(false, errMsg.str());

        return;
    }
}

#endif
