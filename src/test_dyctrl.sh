python test_dyctrl.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --controlnet_model_name_or_path="results/checkpoint" \
    --condition_image="images/seg1.png" \
    --caption="A stone building surrounded by a stone wall and a grassy lawn" \
    --num_sampling_steps=20 \
    --save_generated_image="image/generated_image" \
