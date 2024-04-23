source /raid/s2198939/miniconda3/bin/activate demm
cd /raid/s2198939/diffusion_memorization


pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
train_batch_size=64
max_train_steps=100
checkpointing_steps=10000
validation_steps=100
resolution=224
learning_rate=1e-5
lr_warmup_steps=0
gradient_accumulation_steps=1
unet_pretraining_type="full"
output_dir="OUTPUT_HPO"

num_trials=1
n_repeats=300
data_size_ratio=0.1
objective_metric="max_norm"

CUDA_VISIBLE_DEVICES=6

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_text_to_image_HPO.py --mixed_precision "fp16" --output_dir $output_dir --pretrained_model_name_or_path $pretrained_model_name_or_path --resolution $resolution --center_crop --train_batch_size $train_batch_size --gradient_checkpointing --max_train_steps $max_train_steps --learning_rate $learning_rate --max_grad_norm 1 --lr_scheduler="constant" --lr_warmup_steps 0 --unet_pretraining_type $unet_pretraining_type --num_trials $num_trials --validation_steps $validation_steps --n_repeats $n_repeats --data_size_ratio $data_size_ratio --objective_metric $objective_metric