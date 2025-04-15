#!/bin/bash

source /workspace/.miniconda3/etc/profile.d/conda.sh ##change this to your miniconda/conda path


dataset_name="infovqa_val"   ## put any dataset here
#note that the script below can be coded as a loop, however we keep it in seperate parts for illustration purposes
# Set environment variables
export HUGGING_FACE_HUB_TOKEN="put your Hugging Face Token here"
export HF_HOME="$HOME/.cache/huggingface" ##change this if necessary
cd "$(dirname "$(pwd)")" || exit 1
conda activate pactenv

##custom_clustering
## we note that proportioanl attention is activated per default, set no_proportional_attention to false in the config file to disable it
export pact_config_path="configs/custom_clustering.json"
export log_output_path="logs/custom_clustering_llavaonevision_7b"
export cutoff=3 # This overrides config parameters at runtime which prevents the need for multiple config files for different hyperparameters

echo "Running custom_clustering on dataset: $dataset_name"

# Run evaluation
python -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,device_map=auto,conv_template=qwen_1_5 \
    --tasks="$dataset_name" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="llava_one_7b_${dataset_name}" \
    --output_path="lmms_eval/logs/"


#Runnig custom_pruning
export pact_config_path="configs/custom_pruning.json"
export log_output_path="logs/custom_pruning_llavaonevision_7b"
export pruning_tokeep_percentage_value=0.353  # This overrides config parameters at runtime, which prevents the need for multiple config files for different hyperparameters

echo "Running custom_pruning on dataset: $dataset_name"

# Run evaluation
python -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,device_map=auto,conv_template=qwen_1_5 \
    --tasks="$dataset_name" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="llava_one_7b_${dataset_name}" \
    --output_path="lmms_eval/logs/"

#Runnig pruning followed by clustering based reduction
## we note that proportioanl attention is activated per default, set no_proportional_attention to false in the config file to disable it
export pact_config_path="configs/custom_combined.json"
export log_output_path="logs/custom_combined_llavaonevision_7b"
export pruning_tokeep_percentage_value=0.7 # This overrides config parameters at runtime, which prevents the need for multiple config files for different hyperparameters
export cutoff=2

echo "Running pruning followed by clustering based reduction on dataset: $dataset_name"

# Run evaluation
python -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,device_map=auto,conv_template=qwen_1_5 \
    --tasks="$dataset_name" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="llava_one_7b_${dataset_name}" \
    --output_path="lmms_eval/logs/"

