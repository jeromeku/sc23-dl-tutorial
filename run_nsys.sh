#!/usr/bin/bash

PYTHON_EXE=`which torchrun`
TIMESTAMP=`date +%Y%m%d%H%M`
TRACE_DIR="nsys_profs"
PROF_PREFIX="nsys.prof%p"

# DATA_CONFIG=$1
# MODEL_CONFIG=$2

# DATA_CONFIG=${DATA_CONFIG:-"data/opengenome.yml"}
# MODEL_CONFIG=${MODEL_CONFIG:-"model/evo2/7b_test.yml"}

# echo "DATA_CONFIG is set to: $DATA_CONFIG"
# echo "MODEL_CONFIG is set to: $MODEL_CONFIG"


GENERATED_PROF_NAME="$PROF_PREFIX.nsys-rep"
if [ ! -d "$TRACE_DIR/$TIMESTAMP" ]; then
    mkdir -p $TRACE_DIR/$TIMESTAMP    
fi

PROFILE_CMD="nsys profile \
--gpu-metrics-device=all \
--cuda-memory-usage=true \
--cudabacktrace=true \
--python-sampling=true \
--capture-range=cudaProfilerApi \
--stats=false \
-w true \
-t cuda,nvtx,osrt,cudnn,cublas-verbose \
-s process-tree \
-o $TRACE_DIR/$TIMESTAMP/$PROF_PREFIX \
-f true \
-x true \
$PYTHON_EXE --nproc_per_node=2 --nnode=1 train.py --config=short"

#Not used currently -- this will generate sqlite db from nsys-rep which can be time consuming and memory intensive
PRINT_STATS_CMD="nsys stats $TRACE_DIR/$TIMESTAMP/$GENERATED_PROF_NAME"
SAVE_STATS_CMD="nsys stats --format=csv --output=$TRACE_DIR/$TIMESTAMP/stats $TRACE_DIR/$TIMESTAMP/$GENERATED_PROF_NAME"

CMD="$PROFILE_CMD" 
# && $PRINT_STATS_CMD" #&& $SAVE_STATS_CMD"
echo $PROFILE_CMD

eval $CMD 2>&1 | tee $TRACE_DIR/$TIMESTAMP/trace.log

#Run nsys recipe --help locally to see available additional stats reports
# E.g., nsys recipe nvtx_gpu_proj_trace --input nsys.prof.nsys-rep 
# nsys stats --format=table --report nvtx_pushpop_trace.py 202408210826.prof.nsys-rep