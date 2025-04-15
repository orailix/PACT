#!/bin/bash

source /workspace/.miniconda3/etc/profile.d/conda.sh ##change this to your miniconda/conda path

dataset_name_list=("mmmu_val" "infovqa_val" "mme" "mmstar"  "docvqa_val" "mmbench_en_dev"   "ai2d" "chartqa"  "muirbench" "llava_interleave_bench_out_domain" "textvqa_val" "egoschema" ) 
# you can add the following datasets if you already put your Open AI/Reka API keys :
#'mmvet' 'mathvista_testmini' 'mathverse_testmini_vision_only' 'live_bench_2406' 'llava_wilder_small' "vibe_eval"

# Set environment variables
export OPENAI_API_TIMEOUT=300
export OPENAI_API_KEY="Put API key here"  #only necessary for the commented datasets above
export REKA_API_KEY="Put API key here" #only necessary for the commented datasets above
export HUGGING_FACE_HUB_TOKEN="put your Hugging Face Token here"
export HF_HOME="$HOME/.cache/huggingface"  ##change this if necessary


cd "$(dirname "$(pwd)")" || exit 1

export pact_config_path="configs/pact.json"
export log_output_path="logs/pact_Qwenvl2.0_7b"
export cutoff=0.1 # This overrides config parameters at runtime, which prevents the need for multiple config files for different hyperparameters
export pruning_tokeep_percentage_value=0.8  # This overrides config parameters at runtime, which prevents the need for multiple config files for different hyperparameters
conda activate pactenv

for dataset_name in "${dataset_name_list[@]}"; do
    echo "Running on dataset: $dataset_name"
    python -m lmms_eval     --model=qwen2_vl     --model_args=pretrained=Qwen/Qwen2-VL-7B-Instruct,device_map=auto   --tasks="$dataset_name"	--batch_size=1     --log_samples     --log_samples_suffix="qwenvl2.0_${dataset_name}"     --output_path="lmms_eval/logs/pact_Qwenvl2.0_7b"
done
