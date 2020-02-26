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

#ifndef _FROSCH_PARTITIONOFUNITY_DECL_HPP
#define _FROSCH_PARTITIONOFUNITY_DECL_HPP

#include <FROSch_DDInterface_def.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC = double,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
              class LO = int,
              class GO = DefaultGlobalOrdinal,
#endif
              class NO = KokkosClassic::DefaultNode::DefaultNodeType>
    class PartitionOfUnity {

    protected:

#ifndef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using LO = typename Tpetra::Map<>::local_ordinal_type;
        using GO = typename Tpetra::Map<>::global_ordinal_type;
#endif
        using CommPtr                       = RCP<const Comm<int> >;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMap                          = Map<LO,GO,NO>;
#else
        using XMap                          = Map<NO>;
#endif
        using XMapPtr                       = RCP<XMap>;
        using ConstXMapPtr                  = RCP<const XMap>;
        using XMapPtrVecPtr                 = ArrayRCP<XMapPtr>;
        using ConstXMapPtrVecPtr            = ArrayRCP<ConstXMapPtr>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMatrix                       = Matrix<SC,LO,GO,NO>;
#else
        using XMatrix                       = Matrix<SC,NO>;
#endif
        using XMatrixPtr                    = RCP<XMatrix>;
        using ConstXMatrixPtr               = RCP<const XMatrix>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMultiVector                  = MultiVector<SC,LO,GO,NO>;
#else
        using XMultiVector                  = MultiVector<SC,NO>;
#endif
        using ConstXMultiVectorPtr          = RCP<const XMultiVector>;
        using XMultiVectorPtr               = RCP<XMultiVector>;
        using XMultiVectorPtrVecPtr         = ArrayRCP<XMultiVectorPtr>;
        using ConstXMultiVectorPtrVecPtr    = ArrayRCP<ConstXMultiVectorPtr>;

        using ParameterListPtr              = RCP<ParameterList>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using DDInterfacePtr                = RCP<DDInterface<SC,LO,GO,NO> >;
        using ConstDDInterfacePtr           = RCP<const DDInterface<SC,LO,GO,NO> >;
#else
        using DDInterfacePtr                = RCP<DDInterface<SC,NO> >;
        using ConstDDInterfacePtr           = RCP<const DDInterface<SC,NO> >;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using EntitySetPtr                  = RCP<EntitySet<SC,LO,GO,NO> >;
#else
        using EntitySetPtr                  = RCP<EntitySet<SC,NO> >;
#endif
        using EntitySetPtrVecPtr            = ArrayRCP<EntitySetPtr>;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using InterfaceEntityPtr            = RCP<InterfaceEntity<SC,LO,GO,NO> >;
#else
        using InterfaceEntityPtr            = RCP<InterfaceEntity<SC,NO> >;
#endif

        using UN                            = unsigned;
        using ConstUN                       = const UN;
        
        using LOVec                         = Array<LO>;
        using LOVecPtr                      = ArrayRCP<LO>;
        using LOVecPtr2D                    = ArrayRCP<LOVecPtr>;
        
        using GOVec                         = Array<GO>;
        using GOVecView                     = ArrayView<GO>;

        using SCVec                         = Array<SC>;
        using SCVecPtr                      = ArrayRCP<SC>;

    public:

        PartitionOfUnity(CommPtr mpiComm,
                         CommPtr serialComm,
                         UN dofsPerNode,
                         ConstXMapPtr nodesMap,
                         ConstXMapPtrVecPtr dofsMaps,
                         ParameterListPtr parameterList,
                         Verbosity verbosity = All,
                         UN levelID = 1);

        virtual ~PartitionOfUnity();

        virtual int removeDirichletNodes(GOVecView dirichletBoundaryDofs,
                                         ConstXMultiVectorPtr nodeList = null) = 0;

        virtual int computePartitionOfUnity(ConstXMultiVectorPtr nodeList = null) = 0;

        int assembledPartitionOfUnityMaps();
        
        XMultiVectorPtrVecPtr getLocalPartitionOfUnity() const;

        XMapPtrVecPtr getPartitionOfUnityMaps() const;
        
        XMapPtr getAssembledPartitionOfUnityMap() const;

    protected:

        CommPtr MpiComm_;
        CommPtr SerialComm_;

        ParameterListPtr ParameterList_;

        XMultiVectorPtrVecPtr LocalPartitionOfUnity_;

        XMapPtrVecPtr PartitionOfUnityMaps_;
        
        XMapPtr AssmbledPartitionOfUnityMap_;

        bool Verbose_ = false;
        
        Verbosity Verbosity_ = All;

        const UN LevelID_;
    };

}

#endif
