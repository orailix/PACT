#!/bin/bash

source /workspace/.miniconda3/etc/profile.d/conda.sh ##change this to your miniconda/conda path


dataset_name="infovqa_val"   ## put any dataset here
#note that the script below can be coded as a loop, however we keep it in seperate parts for illustration purposes
# Set environment variables
export HUGGING_FACE_HUB_TOKEN="put your Hugging Face Token here"
export HF_HOME="$HOME/.cache/huggingface" ##change this if necessary
cd "$(dirname "$(pwd)")" || exit 1
conda activate pactenv

##PACT
export pact_config_path="configs/pact.json"
export log_output_path="logs/pact_llavaonevision_7b"
export cutoff=0.21 # This overrides config parameters at runtime — prevents the need for multiple config files for different hyperparameters
export pruning_tokeep_percentage_value=0.55  # This overrides config parameters at runtime — prevents the need for multiple config files for different hyperparameters

echo "Running PACT on dataset: $dataset_name"

# Run evaluation
python -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,device_map=auto,conv_template=qwen_1_5 \
    --tasks="$dataset_name" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="llava_one_7b_${dataset_name}" \
    --output_path="lmms_eval/logs/"


#Runnig DBDPC
export pact_config_path="configs/dbdpc.json"
export log_output_path="logs/dbdpc_llavaonevision_7b"
export cutoff=0.21 # This overrides config parameters at runtime — prevents the need for multiple config files for different hyperparameters

echo "Running DBDPC on dataset: $dataset_name"

# Run evaluation
python -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,device_map=auto,conv_template=qwen_1_5 \
    --tasks="$dataset_name" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="llava_one_7b_${dataset_name}" \
    --output_path="lmms_eval/logs/"

#Runnig EUTI
export pact_config_path="configs/euti.json"
export log_output_path="logs/euti_llavaonevision_7b"
export pruning_tokeep_percentage_value=0.4 # This overrides config parameters at runtime — prevents the need for multiple config files for different hyperparameters

echo "Running EUTI on dataset: $dataset_name"

# Run evaluation
python -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,device_map=auto,conv_template=qwen_1_5 \
    --tasks="$dataset_name" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="llava_one_7b_${dataset_name}" \
    --output_path="lmms_eval/logs/"



#FastV
export pact_config_path="configs/fastv.json"
export log_output_path="logs/fastv_llavaonevision_7b"
export pruning_tokeep_percentage_value=0.353 # This overrides config parameters at runtime — prevents the need for multiple config files for different hyperparameters

echo "Running FastV on dataset: $dataset_name"

# Run evaluation
python -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,device_map=auto,conv_template=qwen_1_5 \
    --tasks="$dataset_name" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="llava_one_7b_${dataset_name}" \
    --output_path="lmms_eval/logs/"



#TOME
export pact_config_path="configs/tome.json"
export log_output_path="logs/tome_llavaonevision_7b"
export perc_tokeep_tome_total=0.353 # This overrides config parameters at runtime — prevents the need for multiple config files for different hyperparameters
 
echo "Running TOME on dataset: $dataset_name"

# Run evaluation
python -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,device_map=auto,conv_template=qwen_1_5 \
    --tasks="$dataset_name" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="llava_one_7b_${dataset_name}" \
    --output_path="lmms_eval/logs/"


#Vistal Token Withdrawel
export pact_config_path="configs/vtw.json"
export log_output_path="logs/vtw_llavaonevision_7b"
export equivalent_reduc_percentage_vtw=0.353 # This overrides config parameters at runtime — prevents the need for multiple config files for different hyperparameters

echo "Running VTW on dataset: $dataset_name"

# Run evaluation
python -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,device_map=auto,conv_template=qwen_1_5 \
    --tasks="$dataset_name" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="llava_one_7b_${dataset_name}" \
    --output_path="lmms_eval/logs/"

#Runnig DPC
export pact_config_path="configs/dpc.json"
export log_output_path="logs/dpc_llavaonevision_7b"
export percentage_to_keep_dpc=0.4 # This overrides config parameters at runtime — prevents the need for multiple config files for different hyperparameters

echo "Running DPC on dataset: $dataset_name"

# Run evaluation
python -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,device_map=auto,conv_template=qwen_1_5 \
    --tasks="$dataset_name" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="llava_one_7b_${dataset_name}" \
    --output_path="lmms_eval/logs/"
    
#K-means
export pact_config_path="configs/kmeans.json"
export log_output_path="logs/kmeans_llavaonevision_7b"
export perc_tokeep_kmeans=0.4 # This overrides config parameters at runtime — prevents the need for multiple config files for different hyperparameters

echo "Running K-means on dataset: $dataset_name"

# Run evaluation
python -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,device_map=auto,conv_template=qwen_1_5 \
    --tasks="$dataset_name" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="llava_one_7b_${dataset_name}" \
    --output_path="lmms_eval/logs/"




#agglomerative clustering
export pact_config_path="configs/agglomerative.json"
export log_output_path="logs/agglomerative_llavaonevision_7b"
export percentage_to_keep_agglomerative=0.4 # This overrides config parameters at runtime — prevents the need for multiple config files for different hyperparameters

echo "Running  agglomerative clustering on dataset: $dataset_name"

# Run evaluation
python -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,device_map=auto,conv_template=qwen_1_5 \
    --tasks="$dataset_name" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="llava_one_7b_${dataset_name}" \
    --output_path="lmms_eval/logs/"


#DBSCAN clustering
export pact_config_path="configs/dbscan.json"
export log_output_path="logs/DBSCAN_llavaonevision_7b"
export eps_dbscan=0.7 # This overrides config parameters at runtime — prevents the need for multiple config files for different hyperparameters

echo "Running DBSCAN clustering on dataset: $dataset_name"

# Run evaluation
python -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,device_map=auto,conv_template=qwen_1_5 \
    --tasks="$dataset_name" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="llava_one_7b_${dataset_name}" \
    --output_path="lmms_eval/logs/"
