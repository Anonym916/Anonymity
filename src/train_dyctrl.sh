accelerate launch \
    --config_file "yaml/accelerate_config.yaml" \
    train_dyctrl.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --output_dir="results/" \
    --dataset_name="limingcv/MultiGen-20M_depth" \
    --conditioning_image_column=control_depth \
    --image_column=image \
    --caption_column=text \
    --resolution=512 \
    --learning_rate=1e-5 \
    --max_train_steps=100000 \
    --validation_steps=100 \
    --validation_image "cond_adr/image1" \
                       "cond_adr/image2" \
                       "cond_adr/image3" \
    --validation_prompt "prompt1" \
                        "prompt2" \
                        "prompt3" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=32 \
    --checkpointing_steps=1000 \
    --enable_xformers_memory_efficient_attention \
    --resume_from_checkpoint="latest" \
    --report_to wandb \
