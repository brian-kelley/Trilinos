#
# Load the ATDM Trilinos env given the build name.
#
# Usage:
#
#   source <this-dir>/load-tril-env.sh <build-name>
#
# This will make any needed substitutions in <build-name> (like 'gcc' =>
# 'gnu') and then load the matching ATDM Trilinos configuration:
#
#   source <tri-dir>/cmake/std/atdm/load-env.sh <build-name-mod>
#
# where <tri-dir> is <this-dir> unless the env var TRILINOS_REPO_DIR is set.
# The latter allows pointing to different ATDM Trilinos configuration and
# version of Trilinos.
#

# Get the base source Trilinos dir (allowing override)
THIS_TRILINOS_DIR=$(readlink -f $BASH_SOURCE | sed "s/\(.*\)\/.*\.sh/\1/g")
TRILINOS_DIR=${TRILINOS_REPO_DIR:-${THIS_TRILINOS_DIR}}

# Get build-name
build_name=$1 ; shift

# Replace 'gcc' with 'gnu'
build_name_mod="${build_name//gcc/gnu}"

# Load ATDM Trilinos configuration
source ${TRILINOS_DIR}/cmake/std/atdm/load-env.sh ${build_name_mod} "$@"

# NOTE: Above, we pass in any extra arguments passed to this script after the
# first 'build_name' argument.  This can allow passing in a custom
# configuration but it will also catch if more than just two arguments are
# passed in.
