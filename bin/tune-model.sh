#!/usr/bin/env bash

#!/bin/sh

export MAX_EPOCHS=2
export EVAL_FREQ=2

OUT_LOG=results/hyperparams/
mkdir -p $OUT_LOG
echo "Writing to "$OUT_LOG

# run on all available gpus
#gpus=`nvidia-smi -L | wc -l`
gpuids=( 0 1 2 3 )
num_gpus=${#gpuids[@]}

# grid search over these
lrs="0.001 0.01 0.1"
lr_decays="0.1 0.01"
dropouts="0.1"
clipgrads="100" # 0.1 0.25"
word_dims="10 25 50"
hidden_dims="10 25 50"
batchsizes="1024 256"

RUN_CMD="python src/tf/ds-classifer.py "

# array to hold all the commands we'll distribute
declare -a commands

# first make all the commands we want
for word_dim in $word_dims
do
   for lr in $lrs
   do
       for decay in $lr_decays
       do
           for hidden_dim in $hidden_dims
           do
               for batchsize in $batchsizes;
               do
                   for clipgrad in $clipgrads;
                   do
                       for dropout in $dropouts;
                       do
                           RUN_NAME=maxpool-$word_dim-$hidden_dim-$lr-$decay-$dropout-$clipgrad-$batchsize
                           CMD="$RUN_CMD \
                                --word_dim $word_dim \
                                --hidden_dim $hidden_dim \
                                --lr $lr \
                                --lr_decay $decay \
                                --batch_size $batchsize \
                                --dropout $dropout \
                                --max_grad_norm $clipgrad \
                                --result_dir $OUT_LOG/$RUN_NAME \
                                --gpuid 0 \
                                --max_epoch $MAX_EPOCHS \
                                --tac_eval_freq $EVAL_FREQ \
                                --pool \
                                --max \
                                &> $OUT_LOG/train-$RUN_NAME.log"
                           commands+=("$CMD")
                           echo "Adding job $RUN_NAME"
                        done
                    done
               done
           done
       done
   done
done

# now distribute them to the gpus
#
# currently this is only correct if the number of jobs is a 
# multiple of the number of gpus (true as long as you have hyperparams
# ranging over 2, 3 and 4 values)!
num_jobs=${#commands[@]}
jobs_per_gpu=$((num_jobs / num_gpus))
echo "Distributing $num_jobs jobs to $num_gpus gpus ($jobs_per_gpu jobs/gpu)"

j=0
for gpuid in ${gpuids[@]}; do
    for (( i=0; i<$jobs_per_gpu; i++ )); do
        jobid=$((j * jobs_per_gpu + i))
        comm="${commands[$jobid]/XX/$gpuid}"
        echo "Starting job $jobid on gpu $gpuid"
        if [ "$gpuid" -ge 0 ]; then
            comm="CUDA_VISIBLE_DEVICES=$gpuid $comm"
        fi
        eval ${comm}
    done &
    j=$((j + 1))
done
