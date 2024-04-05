#! /bin/bash


llava_path="/home/leikel/junchi/pretrained_weights/LLaVA-lightning-7B-v1/"
vision_path="/home/leikel/junchi/pretrained_weights/SAM/sam_vit_h_4b8939.pth"
dataset_path="/home/leikel/junchi/lisa_dataset"
sam_masks_path="/home/leikel/junchi/processed_data"
log_path="/home/leikel/junchi/lisa_dataset/new_runs"

deepspeed --include localhost:0,1 \
  --master_port=24353 validate_llmseg.py \
  --version="$llava_path" \
  --dataset_dir="$dataset_path" \
  --vision_pretrained="$vision_path" \
  --dataset="reason_seg" \
  --sample_rates="1" \
  --exp_name="finetune_llmseg" \
  --log_base_dir="$log_path" \
  --batch_size=1 \
  --eval_only \
  --visualize \