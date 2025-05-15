#!/bin/bash

set -e
set -o pipefail

model_base=lmsys/vicuna-7b-v1.5
model_path=microsoft/llava-rad

model_base="${1:-$model_base}"
model_path="${2:-$model_path}"
prediction_dir="${3:-results/llavarad}"
prediction_file=$prediction_dir/test

run_name="${4:-llavarad}"


query_file=/data/mariammaa/llava_rad/llava-rad-mimic-cxr-annotation/1.0.0/chat_test_MIMIC_CXR_sample.json

image_folder=/data/mariammaa/llava_rad/llava-rad-mimic-cxr-annotation/1.0.0/files
loader="mimic_test_findings"
conv_mode="v1"

CUDA_VISIBLE_DEVICES=1 python -W ignore -m llava.eval.model_mimic_cxr \
    --query_file ${query_file} \
    --loader ${loader} \
    --image_folder ${image_folder} \
    --conv_mode ${conv_mode} \
    --prediction_file ${prediction_file}_${idx}.jsonl \
    --temperature 0 \
    --model_path ${model_path} \
    --model_base ${model_base} \
    --batch_size 1 \
    --group_by_length &


# wait

# cat ${prediction_file}_*.jsonl > mimic_cxr_preds.jsonl

# pushd llava/eval/rrg_eval
# WANDB_PROJECT="llava" WANDB_RUN_ID="llava-eval-$(date +%Y%m%d%H%M%S)" WANDB_RUN_GROUP=evaluate CUDA_VISIBLE_DEVICES=0 \
#     python run.py ../../../mimic_cxr_preds.jsonl --run_name ${run_name} --output_dir ../../../${prediction_dir}/eval
# popd

# rm mimic_cxr_preds.jsonl