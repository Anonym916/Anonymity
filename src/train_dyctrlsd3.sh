accelerate launch \
    --config_file "yaml/accelerate_config_zero2.yaml" \
    train_dysd3_by_local_cache.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers" \
    --dataset_adr="latent/llava558k/splite_0"
    --output_dir="results" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --max_train_steps=50000 \
    --validation_image "cond_adr/image1" \
                       "cond_adr/image2" \
                       "cond_adr/image3" \
    --validation_prompt "prompt1" \
                        "prompt2" \
                        "prompt3" \
    --validation_steps=100 \
    --train_batch_size=16 \
    --gradient_accumulation_steps=8 \
    --checkpointing_steps=100 \
    --resume_from_checkpoint="latest" \
    --mixed_precision=fp16 \
    --report_to wandb \
