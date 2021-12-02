if   [[ ${1} == 'opt' || ${2} == 'opt' || ${3} == 'opt' || ${4} == 'opt' || ${5} == 'opt' ]]
then
  VARIANT=opt
elif [[ ${1} == 'dbg' || ${2} == 'dbg' || ${3} == 'dbg' || ${4} == 'dbg' || ${5} == 'dbg' ]]
then
  VARIANT=dbg
elif [[ ${1} == 'asan' || ${2} == 'asan' || ${3} == 'asan' || ${4} == 'asan' || ${5} == 'asan' ]]
then
  VARIANT=asan
else
  echo " *** You may specify 'opt', 'dbg', or 'asan' to this configuration script. Defaulting to '${DEFAULT_VARIANT:?}'! ***"
  VARIANT=${DEFAULT_VARIANT:?}
fi

if   [[ ${1} == 'static' || ${2} == 'static' || ${3} == 'static' || ${4} == 'static' || ${5} == 'static' ]]
then
  LINKTYPE=static
  RDCON=nordc
elif [[ ${1} == 'static-rdc' || ${2} == 'static-rdc' || ${3} == 'static-rdc' || ${4} == 'static-rdc' || ${5} == 'static-rdc' ]]
then
  LINKTYPE=static
  RDCON=rdc
elif [[ ${1} == 'shared' || ${2} == 'shared' || ${3} == 'shared' || ${4} == 'shared' || ${5} == 'shared' ]]
then
  LINKTYPE=shared
  RDCON=nordc
else
  echo " *** You may specify 'static' or 'shared' to this configuration script. Defaulting to '${DEFAULT_LINKTYPE:?}'!"
  LINKTYPE=${DEFAULT_LINKTYPE:?}
  RDCON=nordc
fi

if   [[ ${1} == 'serial' || ${2} == 'serial' || ${3} == 'serial' || ${4} == 'serial' || ${5} == 'serial' ]]
then
  EXECUTIONSPACE=serial
elif [[ ${1} == 'openmp' || ${2} == 'openmp' || ${3} == 'openmp' || ${4} == 'openmp' || ${5} == 'openmp' ]]
then
  EXECUTIONSPACE=openmp
else
  echo " *** You may specify 'serial' or 'openmp' to this configuration script. Defaulting to '${DEFAULT_EXECUTIONSPACE:?}'!"
  EXECUTIONSPACE=${DEFAULT_EXECUTIONSPACE:?}
fi

if   [[ ${1} == 'full' || ${2} == 'full' || ${3} == 'full' || ${4} == 'full' || ${5} == 'full' ]]
then
  PACKAGE=full
elif [[ ${1} == 'mini' || ${2} == 'mini' || ${3} == 'mini' || ${4} == 'mini' || ${5} == 'mini' ]]
then
  PACKAGE=mini
elif [[ ${1} == 'seacas' || ${2} == 'seacas' || ${3} == 'seacas' || ${4} == 'seacas' || ${5} == 'seacas' ]]
then
  PACKAGE=seacas
else
  echo " *** You may specify 'full', 'mini', or 'seacas' to this configuration script. Defaulting to '${DEFAULT_PACKAGE:?}'!"
  PACKAGE=${DEFAULT_PACKAGE:?}
fi

if [[ ${1} == 'mpi' || ${2} == 'mpi' || ${3} == 'mpi' || ${4} == 'mpi' || ${5} == 'mpi' ]]
then
  USE_MPI=mpi
elif [[ ${1} == 'nompi' || ${2} == 'nompi' || ${3} == 'nompi' || ${4} == 'nompi' || ${5} == 'nompi' ]]
then
  USE_MPI=nompi
else
  echo " *** You may specify 'mpi' or 'nompi' to this configuration script. Defaulting to '${DEFAULT_USE_MPI:?}'!"
  USE_MPI=${DEFAULT_USE_MPI:?}
fi
