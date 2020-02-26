#ifndef NOX_TPETRA_ME_HEQ_DEF_HPP
#define NOX_TPETRA_ME_HEQ_DEF_HPP

// Teuchos support
#include "Teuchos_CommHelpers.hpp"

// Thyra support
#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"

// Kokkos support
#include "Kokkos_Core.hpp"

#include "Heq_Functors.hpp"

// Nonmember constuctors

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
Teuchos::RCP<EvaluatorTpetraHeq<Scalar, LO, GO, Node> >
#else
template<class Scalar, class Node>
Teuchos::RCP<EvaluatorTpetraHeq<Scalar, Node> >
#endif
evaluatorTpetraHeq(const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
                   const Tpetra::global_size_t numGlobalElements,
                   const Scalar omega)
{
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  return Teuchos::rcp(new EvaluatorTpetraHeq<Scalar, LO, GO, Node>(comm, numGlobalElements, omega));
#else
  return Teuchos::rcp(new EvaluatorTpetraHeq<Scalar, Node>(comm, numGlobalElements, omega));
#endif
}

// Constructor

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
EvaluatorTpetraHeq<Scalar, LO, GO, Node>::
#else
template<class Scalar, class Node>
EvaluatorTpetraHeq<Scalar, Node>::
#endif
EvaluatorTpetraHeq(const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
                   const Tpetra::global_size_t numGlobalElements,
                   const Scalar omega) :
  comm_(comm),
  numGlobalElements_(numGlobalElements),
  omega_(omega),
  showGetInvalidArg_(false)
{
  TEUCHOS_ASSERT(nonnull(comm_));

  // solution space
  GO indexBase = 0;
  xMap_ = Teuchos::rcp(new const tpetra_map(numGlobalElements_, indexBase, comm_));
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  xSpace_ = Thyra::createVectorSpace<Scalar, LO, GO, Node>(xMap_);
#else
  xSpace_ = Thyra::createVectorSpace<Scalar, Node>(xMap_);
#endif

  // residual space
  fMap_ = xMap_;
  fSpace_ = xSpace_;

  // communicate the number of unknows and min GID on each proc to all procs
  procNumElements_.resize(comm_->getSize());
  const std::size_t numMyElements = xMap_->getNodeNumElements();
  Teuchos::gatherAll(*comm_, 1, &numMyElements, comm_->getSize(), procNumElements_.data());

  procMinGIDs_.resize(comm->getSize());
  const GO myMinGID = xMap_->getMinGlobalIndex();
  Teuchos::gatherAll(*comm_, 1, &myMinGID, comm_->getSize(), procMinGIDs_.data());

  // allocate vector for storing the result of the integral operator application
  integralOp_ = Teuchos::rcp(new tpetra_vec(xMap_));

  // setup in/out args
  typedef Thyra::ModelEvaluatorBase MEB;
  MEB::InArgsSetup<Scalar> inArgs;
  inArgs.setModelEvalDescription(this->description());
  inArgs.setSupports(MEB::IN_ARG_x);
  prototypeInArgs_ = inArgs;

  MEB::OutArgsSetup<Scalar> outArgs;
  outArgs.setModelEvalDescription(this->description());
  outArgs.setSupports(MEB::OUT_ARG_f);
  outArgs.setSupports(MEB::OUT_ARG_W_op);
  outArgs.setSupports(MEB::OUT_ARG_W_prec);
  prototypeOutArgs_ = outArgs;

  x0_ = Thyra::createMember(xSpace_);
  V_S(x0_.ptr(), Teuchos::ScalarTraits<scalar_type>::one());
  nominalValues_ = inArgs;
  nominalValues_.set_x(x0_);

  residTimer_ = Teuchos::TimeMonitor::getNewCounter("Model Evaluator: Residual Evaluation");
  intOpTimer_ = Teuchos::TimeMonitor::getNewCounter("Model Evaluator: Integral Operator Evaluation");
}

// Initializers/Accessors


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
void EvaluatorTpetraHeq<Scalar, LO, GO, Node>::
#else
template<class Scalar, class Node>
void EvaluatorTpetraHeq<Scalar, Node>::
#endif
setShowGetInvalidArgs(bool showGetInvalidArg)
{
  showGetInvalidArg_ = showGetInvalidArg;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
void EvaluatorTpetraHeq<Scalar, LO, GO, Node>::
#else
template<class Scalar, class Node>
void EvaluatorTpetraHeq<Scalar, Node>::
#endif
set_W_factory(const Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factory)
{
  W_factory_ = W_factory;
}


// Public functions overridden from ModelEvaulator


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
#else
template<class Scalar, class Node>
#endif
Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
EvaluatorTpetraHeq<Scalar, LO, GO, Node>::get_x_space() const
#else
EvaluatorTpetraHeq<Scalar, Node>::get_x_space() const
#endif
{
  return xSpace_;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
#else
template<class Scalar, class Node>
#endif
Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
EvaluatorTpetraHeq<Scalar, LO, GO, Node>::get_f_space() const
#else
EvaluatorTpetraHeq<Scalar, Node>::get_f_space() const
#endif
{
  return fSpace_;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
#else
template<class Scalar, class Node>
#endif
Thyra::ModelEvaluatorBase::InArgs<Scalar>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
EvaluatorTpetraHeq<Scalar, LO, GO, Node>::getNominalValues() const
#else
EvaluatorTpetraHeq<Scalar, Node>::getNominalValues() const
#endif
{
  return nominalValues_;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
#else
template<class Scalar, class Node>
#endif
Teuchos::RCP<Thyra::LinearOpBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
EvaluatorTpetraHeq<Scalar, LO, GO, Node>::create_W_op() const
#else
EvaluatorTpetraHeq<Scalar, Node>::create_W_op() const
#endif
{
  Teuchos::RCP<jac_op> W_tpetra = Teuchos::rcp(new jac_op(xMap_, procNumElements_, procMinGIDs_));
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  return Thyra::tpetraLinearOp<Scalar, LO, GO, Node>(fSpace_, xSpace_, W_tpetra);
#else
  return Thyra::tpetraLinearOp<Scalar, Node>(fSpace_, xSpace_, W_tpetra);
#endif
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
#else
template<class Scalar, class Node>
#endif
Teuchos::RCP< Thyra::PreconditionerBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
EvaluatorTpetraHeq<Scalar, LO, GO, Node>::create_W_prec() const
#else
EvaluatorTpetraHeq<Scalar, Node>::create_W_prec() const
#endif
{
  // Create the CrsMatrix
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  typedef Tpetra::CrsGraph<LO, GO, Node> tpetra_graph;
#else
  typedef Tpetra::CrsGraph<Node> tpetra_graph;
#endif
  typedef typename tpetra_graph::local_graph_type::row_map_type::non_const_type row_map_type;
  typedef typename tpetra_graph::local_graph_type::entries_type::non_const_type view_type;
  const std::size_t numMyElements = xMap_->getNodeNumElements();
  row_map_type offsets("row offsets", numMyElements+1);
  view_type indices("column indices", numMyElements);
  GraphSetupFunctor<tpetra_graph> functor(offsets, indices, numMyElements);
  Kokkos::parallel_for("graph setup", numMyElements+1, functor);
  Teuchos::RCP<tpetra_graph> graph
    = Teuchos::rcp(new tpetra_graph(fMap_, xMap_, offsets, indices));
  graph->fillComplete();
  Teuchos::RCP<tpetra_mat> W_tpetra
    = Teuchos::rcp(new tpetra_mat(graph));
  W_tpetra->fillComplete();

  // Create the Thyra preconditioner
  Teuchos::RCP<thyra_op> W_op
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
    = Thyra::tpetraLinearOp<Scalar, LO, GO, Node>(fSpace_, xSpace_, W_tpetra);
#else
    = Thyra::tpetraLinearOp<Scalar, Node>(fSpace_, xSpace_, W_tpetra);
#endif
  Teuchos::RCP<Thyra::DefaultPreconditioner<Scalar> > prec
    = Teuchos::rcp(new Thyra::DefaultPreconditioner<Scalar>);
  prec->initializeRight(W_op);
  return prec;
}

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
#else
template<class Scalar, class Node>
#endif
Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar> >
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
EvaluatorTpetraHeq<Scalar, LO, GO, Node>::get_W_factory() const
#else
EvaluatorTpetraHeq<Scalar, Node>::get_W_factory() const
#endif
{
  return W_factory_;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
#else
template<class Scalar, class Node>
#endif
Thyra::ModelEvaluatorBase::InArgs<Scalar>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
EvaluatorTpetraHeq<Scalar, LO, GO, Node>::createInArgs() const
#else
EvaluatorTpetraHeq<Scalar, Node>::createInArgs() const
#endif
{
  return prototypeInArgs_;
}


// Private functions overridden from ModelEvaulatorDefaultBase


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
#else
template<class Scalar, class Node>
#endif
Thyra::ModelEvaluatorBase::OutArgs<Scalar>
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
EvaluatorTpetraHeq<Scalar, LO, GO, Node>::createOutArgsImpl() const
#else
EvaluatorTpetraHeq<Scalar, Node>::createOutArgsImpl() const
#endif
{
  return prototypeOutArgs_;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
void EvaluatorTpetraHeq<Scalar, LO, GO, Node>::
#else
template<class Scalar, class Node>
void EvaluatorTpetraHeq<Scalar, Node>::
#endif
evalModelImpl(const Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
              const Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs) const
{
  TEUCHOS_ASSERT(nonnull(inArgs.get_x()));

  const Teuchos::RCP<thyra_vec> f_out = outArgs.get_f();
  const Teuchos::RCP<thyra_op> W_out = outArgs.get_W_op();
  const Teuchos::RCP<thyra_prec> W_prec_out = outArgs.get_W_prec();

  const bool fill_f = nonnull(f_out);
  const bool fill_W = nonnull(W_out);
  const bool fill_W_prec = nonnull(W_prec_out);

#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
  typedef Thyra::TpetraOperatorVectorExtraction<Scalar,LO,GO,Node> tpetra_extract;
#else
  typedef Thyra::TpetraOperatorVectorExtraction<Scalar,Node> tpetra_extract;
#endif

  if ( fill_f || fill_W || fill_W_prec ) {

    // Get the underlying tpetra objects
    Teuchos::RCP<const tpetra_vec> u = tpetra_extract::getConstTpetraVector(inArgs.get_x());

    Teuchos::RCP<tpetra_vec> f;
    if (fill_f) {
      f = tpetra_extract::getTpetraVector(f_out);
    }

    Teuchos::RCP<jac_op> J;
    if (fill_W) {
      Teuchos::RCP<tpetra_op> M_tpetra = tpetra_extract::getTpetraOperator(W_out);
      J = Teuchos::rcp_dynamic_cast<jac_op>(M_tpetra);
      TEUCHOS_ASSERT(nonnull(J));
    }

    Teuchos::RCP<tpetra_mat> M_inv;
    if (fill_W_prec) {
      Teuchos::RCP<tpetra_op> M_tpetra = tpetra_extract::getTpetraOperator(W_prec_out->getNonconstRightPrecOp());
      M_inv = Teuchos::rcp_dynamic_cast<tpetra_mat>(M_tpetra);
      TEUCHOS_ASSERT(nonnull(M_inv));
      M_inv->resumeFill();
    }

    // Zero out the objects that will be filled
    integralOp_->putScalar(Teuchos::ScalarTraits<scalar_type>::zero());
    if (fill_f) {
      f->putScalar(Teuchos::ScalarTraits<scalar_type>::zero());
    }
    if (fill_W) {
      J->unitialize();
    }
    if (fill_W_prec) {
      M_inv->setAllToScalar(Teuchos::ScalarTraits<scalar_type>::zero());
    }

    // Compute the integral operator
    typedef typename tpetra_vec::dual_view_type::t_dev view_type;
    typedef typename tpetra_vec::execution_space execution_space;
    typedef Kokkos::TeamPolicy<execution_space> team_policy;
    Teuchos::rcp_const_cast<tpetra_vec>(u)->template sync<execution_space>();
    integralOp_->template sync<execution_space>();
    integralOp_->template modify<execution_space>();
    const int myRank = comm_->getRank();
    const GO myMinGID = xMap_->getMinGlobalIndex();
    {
    Teuchos::TimeMonitor timer(*intOpTimer_);
    for (int proc = 0; proc < comm_->getSize(); ++proc) {
      // Compute the local reduction contribution on each proc
      const std::size_t procNumElements = procNumElements_[proc];
      const GO procMinGID = procMinGIDs_[proc];
      Teuchos::RCP<tpetra_vec> localSum;
      if (comm_->getSize() == 1)
        localSum = integralOp_;
      else {
        Teuchos::RCP<const tpetra_map> map
          = Teuchos::rcp(new const tpetra_map(procNumElements, 0, comm_, Tpetra::LocallyReplicated));
        localSum = Teuchos::rcp(new tpetra_vec(map));
      }

      IntegralOperatorFunctor<tpetra_vec> functor(*u, *localSum, myMinGID, procMinGID);
      Kokkos::parallel_for("integral operator", team_policy(procNumElements, Kokkos::AUTO), functor);

      // Reduce local contributions and fill the vector
      if (comm_->getSize() > 1) {
        localSum->template modify<execution_space>();
        localSum->reduce();
        if (myRank == proc) {
          view_type source = localSum->template getLocalView<execution_space>();
          view_type target = integralOp_->template getLocalView<execution_space>();
          Kokkos::deep_copy(target, source);
        }
      }
    }
    integralOp_->scale(omega_/static_cast<Scalar>(2*numGlobalElements_));
    Kokkos::fence();
    }

    // Residual computation
    const std::size_t numMyElements = xMap_->getNodeNumElements();
    if (fill_f) {
      Teuchos::TimeMonitor timer(*residTimer_);
      f->template sync<execution_space>();
      f->template modify<execution_space>();

      ResidualEvaluatorFunctor<tpetra_vec> functor(*f, *u, *integralOp_);
      Kokkos::parallel_for("residual evaluation", numMyElements, functor);
      Kokkos::fence();
    }

    // Jacobian operator computation
    if (fill_W) {
      J->initialize(omega_, integralOp_);
    }

    // Preconditioner computation
    if (fill_W_prec) {
      PreconditionerEvaluatorFunctor<tpetra_vec, tpetra_mat> functor(*M_inv, *integralOp_, omega_, numGlobalElements_);
      Kokkos::parallel_for("prec evaluation", numMyElements, functor);
      M_inv->fillComplete();
    }
  }
}


////////////////////////////////////////////////////////
// Jacobian operator implementations
////////////////////////////////////////////////////////


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
HeqJacobianOperator<Scalar, LO, GO, Node>::
HeqJacobianOperator(const Teuchos::RCP<const Tpetra::Map<LO, GO, Node> >& map,
#else
template<class Scalar, class Node>
HeqJacobianOperator<Scalar, Node>::
HeqJacobianOperator(const Teuchos::RCP<const Tpetra::Map<Node> >& map,
#endif
                    const std::vector<std::size_t>& procNumElements,
                    const std::vector<GO>& procMinGIDs) :
  map_(map),
  procNumElements_(procNumElements),
  procMinGIDs_(procMinGIDs)
{
  integralOpX_ = Teuchos::rcp(new tpetra_vec(map_));
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
void HeqJacobianOperator<Scalar, LO, GO, Node>::
#else
template<class Scalar, class Node>
void HeqJacobianOperator<Scalar, Node>::
#endif
initialize(const Scalar& omega,
#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
           const Teuchos::RCP<Tpetra::Vector<Scalar, LO, GO, Node> >& integralOp)
#else
           const Teuchos::RCP<Tpetra::Vector<Scalar, Node> >& integralOp)
#endif
{
  omega_ = omega;

  TEUCHOS_ASSERT(map_->isSameAs(*integralOp->getMap()));
  integralOp_ = integralOp;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
void HeqJacobianOperator<Scalar, LO, GO, Node>::
#else
template<class Scalar, class Node>
void HeqJacobianOperator<Scalar, Node>::
#endif
unitialize()
{
  omega_ = Teuchos::ScalarTraits<scalar_type>::zero();
  integralOp_ = Teuchos::null;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
Teuchos::RCP<const Tpetra::Map<LO, GO, Node> >
HeqJacobianOperator<Scalar, LO, GO, Node>::getDomainMap() const
#else
template<class Scalar, class Node>
Teuchos::RCP<const Tpetra::Map<Node> >
HeqJacobianOperator<Scalar, Node>::getDomainMap() const
#endif
{
  return map_;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
Teuchos::RCP<const Tpetra::Map<LO, GO, Node> >
HeqJacobianOperator<Scalar, LO, GO, Node>::getRangeMap() const
#else
template<class Scalar, class Node>
Teuchos::RCP<const Tpetra::Map<Node> >
HeqJacobianOperator<Scalar, Node>::getRangeMap() const
#endif
{
  return map_;
}


#ifdef TPETRA_ENABLE_TEMPLATE_ORDINALS
template<class Scalar, class LO, class GO, class Node>
void HeqJacobianOperator<Scalar, LO, GO, Node>::
apply(const Tpetra::MultiVector<Scalar,LO,GO,Node>& X,
      Tpetra::MultiVector<Scalar,LO,GO,Node>& Y,
#else
template<class Scalar, class Node>
void HeqJacobianOperator<Scalar, Node>::
apply(const Tpetra::MultiVector<Scalar,Node>& X,
      Tpetra::MultiVector<Scalar,Node>& Y,
#endif
      Teuchos::ETransp mode,
      Scalar alpha,
      Scalar beta) const
{
  TEUCHOS_ASSERT(nonnull(integralOp_));
  TEUCHOS_ASSERT(mode == Teuchos::NO_TRANS);

  Scalar zero = Teuchos::ScalarTraits<Scalar>::zero();
  Scalar one = Teuchos::ScalarTraits<Scalar>::one();
  if (alpha == zero) {
    if (beta == zero)
      Y.putScalar(zero);
    else if (beta != one)
      Y.scale(beta);
    return;
  }

  // Apply the integral operator to the input multivector
  typedef typename tpetra_vec::dual_view_type::t_dev view_type;
  typedef typename tpetra_vec::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> team_policy;
  const_cast<tpetra_mv&>(X).template sync<execution_space>();
  Y.template sync<execution_space>();
  Y.template modify<execution_space>();
  integralOp_->template sync<execution_space>();
  Teuchos::RCP<const Teuchos::Comm<int> > comm = map_->getComm();
  const int myRank = comm->getRank();
  const GO myMinGID = map_->getMinGlobalIndex();
  const std::size_t numMyElements = map_->getNodeNumElements();

  // Loop over vecs
  for (std::size_t col = 0; col < X.getNumVectors(); ++col) {
    integralOpX_->template sync<execution_space>();
    integralOpX_->template modify<execution_space>();
    integralOpX_->putScalar(zero);

    // Loop over sections of vec
    for (int proc = 0; proc < comm->getSize(); ++proc) {
      const std::size_t procNumElements = procNumElements_[proc];
      const GO procMinGID = procMinGIDs_[proc];
      Teuchos::RCP<tpetra_vec> localResult;
      if (comm->getSize() == 1)
        localResult = integralOpX_;
      else {
        Teuchos::RCP<const tpetra_map> map
          = Teuchos::rcp(new const tpetra_map(procNumElements, 0, comm, Tpetra::LocallyReplicated));
        localResult = Teuchos::rcp(new tpetra_vec(map));
      }

      IntegralOperatorFunctor<tpetra_vec> functor(*X.getVector(col), *localResult, myMinGID, procMinGID);
      Kokkos::parallel_for("integral operator", team_policy(procNumElements, Kokkos::AUTO), functor);

      if (comm->getSize() > 1) {
        localResult->template modify<execution_space>();
        localResult->reduce();
        if (myRank == proc) {
          view_type source = localResult->template getLocalView<execution_space>();
          view_type target = integralOpX_->template getLocalView<execution_space>();
          Kokkos::deep_copy(target, source);
        }
      }
    }

    // Compute the Jacobian-vector product
    JacobianEvaluatorFunctor<tpetra_vec>
      functor(*Y.getVector(col), *X.getVector(col), *integralOp_, *integralOpX_, alpha, beta, omega_);
    Kokkos::parallel_for("jacobian evaluation", numMyElements, functor);

  }
}

#endif
