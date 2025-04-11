#!/bin/bash

source /workspace/.miniconda3/etc/profile.d/conda.sh ##change this to your miniconda/conda path

dataset_name_list=(
  "mmmu_val" "infovqa_val" "mme" "mmstar" "docvqa_val" "mmbench_en_dev"
  "ai2d" "chartqa" "muirbench" "llava_interleave_bench_out_domain"
  "scienceqa_img" "textvqa_val" "videomme" "mlvu" "perceptiontest_test_mc" "egoschema")
  # you can add the following datasets if you already put your Open AI/Reka API keys : "activitynetqa" "videochatgpt""mmvet" "mathvista_testmini" "mathverse_testmini_vision_only" "live_bench_2406" "llava_wilder_small" "vibe_eval"

  
# Set environment variables
export OPENAI_API_TIMEOUT=300
export OPENAI_API_KEY="Put API key here"  #only necessary for the commented datasets above
export REKA_API_KEY="Put API key here" #only necessary for the commented datasets above
export HUGGING_FACE_HUB_TOKEN="put your Hugging Face Token here"
export HF_HOME="$HOME/.cache/huggingface"  ##change this if necessary

cd "$(dirname "$(pwd)")" || exit 1
export pact_config_path="configs/pact.json"
export log_output_path="logs/pact_llavaonevision_7b"
export cutoff=0.21 # This overrides config parameters at runtime — prevents the need for multiple config files for different hyperparameters
export pruning_tokeep_percentage_value=0.55  # This overrides config parameters at runtime — prevents the need for multiple config files for different hyperparameters
conda activate pactenv

# export HF_HOME="$HOME/.cache/huggingface" #change this if you have another directory
# Loop over datasets
for dataset_name in "${dataset_name_list[@]}"; do
    echo "Running on dataset: $dataset_name"
    python -m lmms_eval \
        --model=llava_onevision \
        --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,device_map=auto,conv_template=qwen_1_5 \
        --tasks="$dataset_name" \
        --batch_size=1 \
        --log_samples \
        --log_samples_suffix="llava_one_7b_${dataset_name}" \
        --output_path="lmms_eval/logs/pact_llavaonevision_7b"
done
