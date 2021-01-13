export KOKKOS_NUM_DEVICES=1#!/bin/bash
bsub -q pbatch -Is -nnodes 1 -W 10 --shared-launch ./runTest.sh

