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

#ifndef _FROSCH_RGDSWCOARSEOPERATOR_DECL_HPP
#define _FROSCH_RGDSWCOARSEOPERATOR_DECL_HPP

#include <FROSch_GDSWCoarseOperator_def.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC = double,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
              class LO = int,
              class GO = DefaultGlobalOrdinal,
#endif
              class NO = KokkosClassic::DefaultNode::DefaultNodeType>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    class RGDSWCoarseOperator : public GDSWCoarseOperator<SC,LO,GO,NO> {
#else
    class RGDSWCoarseOperator : public GDSWCoarseOperator<SC,NO> {
#endif

    protected:

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using CommPtr                 = typename SchwarzOperator<SC,LO,GO,NO>::CommPtr;
#else
        using LO = typename Tpetra::Map<>::local_ordinal_type;
        using GO = typename Tpetra::Map<>::global_ordinal_type;
        using CommPtr                 = typename SchwarzOperator<SC,NO>::CommPtr;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMapPtr                 = typename SchwarzOperator<SC,LO,GO,NO>::XMapPtr;
        using ConstXMapPtr            = typename SchwarzOperator<SC,LO,GO,NO>::ConstXMapPtr;
        using XMapPtrVecPtr           = typename SchwarzOperator<SC,LO,GO,NO>::XMapPtrVecPtr;
        using ConstXMapPtrVecPtr      = typename SchwarzOperator<SC,LO,GO,NO>::ConstXMapPtrVecPtr;
#else
        using XMapPtr                 = typename SchwarzOperator<SC,NO>::XMapPtr;
        using ConstXMapPtr            = typename SchwarzOperator<SC,NO>::ConstXMapPtr;
        using XMapPtrVecPtr           = typename SchwarzOperator<SC,NO>::XMapPtrVecPtr;
        using ConstXMapPtrVecPtr      = typename SchwarzOperator<SC,NO>::ConstXMapPtrVecPtr;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMatrixPtr              = typename SchwarzOperator<SC,LO,GO,NO>::XMatrixPtr;
        using ConstXMatrixPtr         = typename SchwarzOperator<SC,LO,GO,NO>::ConstXMatrixPtr;
#else
        using XMatrixPtr              = typename SchwarzOperator<SC,NO>::XMatrixPtr;
        using ConstXMatrixPtr         = typename SchwarzOperator<SC,NO>::ConstXMatrixPtr;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using XMultiVectorPtr         = typename SchwarzOperator<SC,LO,GO,NO>::XMultiVectorPtr;
        using ConstXMultiVectorPtr    = typename SchwarzOperator<SC,LO,GO,NO>::ConstXMultiVectorPtr;
        using XMultiVectorPtrVecPtr   = typename SchwarzOperator<SC,LO,GO,NO>::XMultiVectorPtrVecPtr;
#else
        using XMultiVectorPtr         = typename SchwarzOperator<SC,NO>::XMultiVectorPtr;
        using ConstXMultiVectorPtr    = typename SchwarzOperator<SC,NO>::ConstXMultiVectorPtr;
        using XMultiVectorPtrVecPtr   = typename SchwarzOperator<SC,NO>::XMultiVectorPtrVecPtr;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using ParameterListPtr        = typename SchwarzOperator<SC,LO,GO,NO>::ParameterListPtr;
#else
        using ParameterListPtr        = typename SchwarzOperator<SC,NO>::ParameterListPtr;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using DDInterfacePtr          = typename SchwarzOperator<SC,LO,GO,NO>::DDInterfacePtr;
#else
        using DDInterfacePtr          = typename SchwarzOperator<SC,NO>::DDInterfacePtr;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using EntitySetPtr            = typename SchwarzOperator<SC,LO,GO,NO>::EntitySetPtr;
        using EntitySetPtrVecPtr      = typename SchwarzOperator<SC,LO,GO,NO>::EntitySetPtrVecPtr;
#else
        using EntitySetPtr            = typename SchwarzOperator<SC,NO>::EntitySetPtr;
        using EntitySetPtrVecPtr      = typename SchwarzOperator<SC,NO>::EntitySetPtrVecPtr;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using InterfaceEntityPtr      = typename SchwarzOperator<SC,LO,GO,NO>::InterfaceEntityPtr;
#else
        using InterfaceEntityPtr      = typename SchwarzOperator<SC,NO>::InterfaceEntityPtr;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using UN                      = typename SchwarzOperator<SC,LO,GO,NO>::UN;
#else
        using UN                      = typename SchwarzOperator<SC,NO>::UN;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using LOVec                   = typename SchwarzOperator<SC,LO,GO,NO>::LOVec;
        using LOVecPtr                = typename SchwarzOperator<SC,LO,GO,NO>::LOVecPtr;
        using LOVecPtr2D              = typename SchwarzOperator<SC,LO,GO,NO>::LOVecPtr2D;
#else
        using LOVec                   = typename SchwarzOperator<SC,NO>::LOVec;
        using LOVecPtr                = typename SchwarzOperator<SC,NO>::LOVecPtr;
        using LOVecPtr2D              = typename SchwarzOperator<SC,NO>::LOVecPtr2D;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using GOVec                   = typename SchwarzOperator<SC,LO,GO,NO>::GOVec;
        using GOVecPtr                = typename SchwarzOperator<SC,LO,GO,NO>::GOVecPtr;
        using GOVecPtr2D              = typename SchwarzOperator<SC,LO,GO,NO>::GOVecPtr2D;
#else
        using GOVec                   = typename SchwarzOperator<SC,NO>::GOVec;
        using GOVecPtr                = typename SchwarzOperator<SC,NO>::GOVecPtr;
        using GOVecPtr2D              = typename SchwarzOperator<SC,NO>::GOVecPtr2D;
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
        using SCVecPtr                = typename SchwarzOperator<SC,LO,GO,NO>::SCVecPtr;
#else
        using SCVecPtr                = typename SchwarzOperator<SC,NO>::SCVecPtr;
#endif

    public:

        RGDSWCoarseOperator(ConstXMatrixPtr k,
                            ParameterListPtr parameterList);

        virtual int resetCoarseSpaceBlock(UN blockId,
                                          UN dimension,
                                          UN dofsPerNode,
                                          ConstXMapPtr nodesMap,
                                          ConstXMapPtrVecPtr dofsMaps,
                                          GOVecPtr dirichletBoundaryDofs,
                                          ConstXMultiVectorPtr nodeList);


    protected:

        virtual XMultiVectorPtrVecPtr computeTranslations(UN blockId,
                                                          EntitySetPtr Roots,
                                                          EntitySetPtrVecPtr entitySetVector,
                                                          DistanceFunction distanceFunction = ConstantDistanceFunction);

        virtual XMultiVectorPtrVecPtr computeRotations(UN blockId,
                                                       UN dimension,
                                                       ConstXMultiVectorPtr nodeList,
                                                       EntitySetPtr Roots,
                                                       EntitySetPtrVecPtr entitySetVector,
                                                       DistanceFunction distanceFunction = ConstantDistanceFunction);
    };

}

#endif
