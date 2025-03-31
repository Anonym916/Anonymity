python test_dyctrl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers" \
    --controlnet_model_name_or_path="results/checkpoint" \
    --condition_image="images/canny2.png" \
    --caption="A beautiful unicorn with a rainbow on a background of clouds unicorn cartoon character for pringting posters." \
    --num_sampling_steps=28 \
    --save_generated_image="image/generated_image" \
    --controlnet_conditioning_scale=0.8 \
    --guidance_scale=4.5 \
    --resolution=512 \
