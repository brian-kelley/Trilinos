#!/bin/bash

export KOKKOS_NUM_DEVICES=1
#export KOKKOS_PROFILE_LIBRARY=/ascldap/users/bmkelle/kokkos-tools/kp_memory_usage.so
#export KOKKOS_PROFILE_LIBRARY=/ascldap/users/bmkelle/kokkos-tools/profiling/memory-hwm-mpi/kp_hwm_mpi.so
export KOKKOS_PROFILE_LIBRARY=/ascldap/users/bmkelle/kokkos-tools/profiling/memory-events/kp_memory_events.so

jsrun -M -gpu -p 4 -n 4 -c 1 -g 1 ./TrilinosCouplings_fenl_pce.exe --fixture=8x8x8 --belos --muelu --mean-based --print-its --unit-test --test-mean=2.0599472 --test-std-dev=8.9771555e-3 --verbose --print-its
