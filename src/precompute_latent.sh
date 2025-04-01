accelerate launch \
    --config_file "yaml/accelerate_config_zero2.yaml" \
    precompute_latent.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers" \
    --output_dir="latent/llava558k/splite_0" \
    --dataset_name="dataset/llava558k" \
    --conditioning_image_column=conditioning_image \
    --image_column=image \
    --caption_column=text \
    --resolution=1024 \
    --learning_rate=1e-5 \
    --max_train_steps=50000 \
    --validation_image "image/image1" \
    --validation_prompt "prompt1" \
    --validation_steps=100 \
    --train_batch_size=16 \
    --gradient_accumulation_steps=8 \
    --checkpointing_steps=100 \
    --resume_from_checkpoint="latest" \
    --mixed_precision=fp16 \
    --report_to wandb \
