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

#ifndef _FROSCH_EXTRACTSUBMATRICES_DECL_HPP
#define _FROSCH_EXTRACTSUBMATRICES_DECL_HPP

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

#ifndef FROSCH_TEST_OUTPUT
#define FROSCH_TEST_OUTPUT(COMM,VERBOSE,OUTPUT) COMM->barrier(); COMM->barrier(); COMM->barrier(); if (VERBOSE) std::cout << OUTPUT << std::endl;
#endif

#include <Xpetra_MapFactory_fwd.hpp>
#include <Xpetra_MatrixFactory_fwd.hpp>
#include <Xpetra_ImportFactory_fwd.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    RCP<const Matrix<SC,LO,GO,NO> > ExtractLocalSubdomainMatrix(RCP<const Matrix<SC,LO,GO,NO> > globalMatrix,
                                                                RCP<const Map<LO,GO,NO> > map);

    template <class SC,class LO,class GO,class NO>
    RCP<const Matrix<SC,LO,GO,NO> > ExtractLocalSubdomainMatrix(RCP<const Matrix<SC,LO,GO,NO> > globalMatrix,
                                                                RCP<const Map<LO,GO,NO> > map,
#else
    template <class SC,class NO>
    RCP<const Matrix<SC,NO> > ExtractLocalSubdomainMatrix(RCP<const Matrix<SC,NO> > globalMatrix,
                                                                RCP<const Map<NO> > map);

    template <class SC,class NO>
    RCP<const Matrix<SC,NO> > ExtractLocalSubdomainMatrix(RCP<const Matrix<SC,NO> > globalMatrix,
                                                                RCP<const Map<NO> > map,
#endif
                                                                SC value);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int UpdateLocalSubdomainMatrix(RCP<Matrix<SC,LO,GO,NO> > globalMatrix,
                                   RCP<Map<LO,GO,NO> > &map,
                                   RCP<Matrix<SC,LO,GO,NO> > &localSubdomainMatrix);
#else
    template <class SC,class NO>
    int UpdateLocalSubdomainMatrix(RCP<Matrix<SC,NO> > globalMatrix,
                                   RCP<Map<NO> > &map,
                                   RCP<Matrix<SC,NO> > &localSubdomainMatrix);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int BuildSubmatrices(RCP<const Matrix<SC,LO,GO,NO> > k,
#else
    template <class SC,class NO>
    int BuildSubmatrices(RCP<const Matrix<SC,NO> > k,
#endif
                         ArrayView<GO> indI,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                         RCP<const Matrix<SC,LO,GO,NO> > &kII,
                         RCP<const Matrix<SC,LO,GO,NO> > &kIJ,
                         RCP<const Matrix<SC,LO,GO,NO> > &kJI,
                         RCP<const Matrix<SC,LO,GO,NO> > &kJJ);
#else
                         RCP<const Matrix<SC,NO> > &kII,
                         RCP<const Matrix<SC,NO> > &kIJ,
                         RCP<const Matrix<SC,NO> > &kJI,
                         RCP<const Matrix<SC,NO> > &kJJ);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class SC,class LO,class GO,class NO>
    int BuildSubmatrix(RCP<const Matrix<SC,LO,GO,NO> > k,
#else
    template <class SC,class NO>
    int BuildSubmatrix(RCP<const Matrix<SC,NO> > k,
#endif
                       ArrayView<GO> indI,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                       RCP<const Matrix<SC,LO,GO,NO> > &kII);
#else
                       RCP<const Matrix<SC,NO> > &kII);
#endif

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    template <class LO,class GO,class NO>
    int BuildSubgraph(RCP<const CrsGraph<LO,GO,NO> > k,
#else
    template <class NO>
    int BuildSubgraph(RCP<const CrsGraph<NO> > k,
#endif
                      ArrayView<GO> indI,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
                      RCP<const CrsGraph<LO,GO,NO> > &kII);
#else
                      RCP<const CrsGraph<NO> > &kII);
#endif
}

#endif
