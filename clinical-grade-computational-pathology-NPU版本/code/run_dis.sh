#!/bin/bash
export HCCL_WHITELIST_DISABLE=1
RANK_ID_START=0
WORLD_SIZE=2
for((RANK_ID=$RANK_ID_START;RANK_ID<$((WORLD_SIZE+RANK_ID_START));RANK_ID++));
do
    echo "Device ID: $RANK_ID"
    export LOCAL_RANK=$RANK_ID
    python3 MIL_train.py &
done
wait 
