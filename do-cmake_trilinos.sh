#!/bin/bash -e

#
# Configure and then build and install Trilinos for SPARC using:
#
#   mkdir <build-dir>/
#   cd <build-dir>/
#   source <trilinos-dir>/load-tril-env.sh <buld-name>
#   <trilinos-dir>/do-cmake_trilinos.sh full \
#     -GNinja \
#     -DCMAKE_INSTALL_PREFIX=<install-prefix>
#   ninja -j16 install
#
# For Mini Trilinos, replace 'full' above with 'mini'.  For just SEACAS,
# replace 'full' with 'seacas'.
#
# To use Makefiles instead of Ninja, just remove '-GNinja' from commandline
# arguments and run 'make' instead of 'ninja'.
#

# Get the base Trilinos directory for the location of this script
THIS_TRILINOS_DIR=$(readlink -f $BASH_SOURCE | sed "s/\(.*\)\/.*\.sh/\1/g")
TRILINOS_DIR=${TRILINOS_REPO_DIR:-${THIS_TRILINOS_DIR}}

# Assert that ATDM Trilinos env is already set
if [[ "${ATDM_CONFIG_COMPLETED_ENV_SETUP}" != "TRUE" ]] ; then
  echo "ERROR, ATDM_CONFIG_COMPLETED_ENV_SETUP is not set to TRUE." \
   " Must 'source ${THIS_TRILINOS_DIR}/load-tril-env.sh <build-name>' first!"
  exit 1
fi

# Get the set of Trilinos packages to enable for SPARC from the first
# command-line argument

set_arg=$1 ; shift
if [[ "${set_arg}" == "full" ]] ; then
  pre_config_option_files="cmake/std/atdm/apps/sparc/SPARCTrilinosPackagesEnables.cmake"
elif [[ "${set_arg}" == "mini" ]] ; then
  pre_config_option_files="cmake/std/atdm/apps/sparc/SPARCMiniTrilinosPackagesEnables.cmake"
elif [[ "${set_arg}" == "seacas" ]] ; then
  echo "TODO: Add support for 'seacas' Trilinos build!"
  exit 1
else
  echo "ERROR: First argument must be 'full', 'mini', or 'seacas'!"
  exit 1
fi

# Configure Trilinos
cmake \
-D Trilinos_CONFIGURE_OPTIONS_FILE:STRING=${pre_config_option_files},cmake/std/atdm/ATDMDevEnv.cmake \
"$@" \
${TRILINOS_DIR}
