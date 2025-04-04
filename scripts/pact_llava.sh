#!/bin/bash

source /workspace/.miniconda3/etc/profile.d/conda.sh ##change this to your miniconda/conda path


dataset_name_list=(
  "mmmu_val" "infovqa_val" "mme" "mmstar" "docvqa_val" "mmbench_en_dev"
  "ai2d" "chartqa" "muirbench" "llava_interleave_bench_out_domain"
  "scienceqa_img" "textvqa_val" "activitynetqa" "videochatgpt"
  "mmvet" "mathvista_testmini" "mathverse_testmini_vision_only"
  "live_bench_2406" "llava_wilder_small" "vibe_eval" "videomme"
  "mlvu" "perceptiontest_test_mc" "egoschema"
)


# Set environment variables
export OPENAI_API_TIMEOUT=300
export OPENAI_API_KEY="put your openai api key here"
export REKA_API_KEY="put your reka api key here"
export HUGGING_FACE_HUB_TOKEN="put your huggingface token here"
export HF_HOME="$HOME/.cache/huggingface" #change this if you have another directory
# Loop over datasets
for dataset_name in "${dataset_name_list[@]}"; do
    echo "Running on dataset: $dataset_name"

    # Activate conda environment
    conda activate pact

    # Set config path (relative or absolute)
    export pact_config_path="$(dirname "$(pwd)")/configs/pact.json"

    # Change to evaluation directory
    cd "$(dirname "$(pwd)")/lmms_eval" || exit 1

    # Run evaluation
    python -m lmms_eval \
        --model=llava_onevision \
        --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,device_map=auto,conv_template=qwen_1_5 \
        --tasks="$dataset_name" \
        --batch_size=1 \
        --log_samples \
        --log_samples_suffix="llava_one_7b_${dataset_name}" \
        --output_path="./logs/"
done
