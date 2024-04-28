source /raid/s2198939/miniconda3/bin/activate demm
cd /raid/s2198939/Diffusion_Memorization_HPO

pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
train_batch_size=128
max_train_steps=3000
checkpointing_steps=10000
resolution=224
learning_rate=1e-4
lr_warmup_steps=0
gradient_accumulation_steps=1
unet_pretraining_type="auto_svdiff"
output_dir="OUTPUT_SVDIFF_HPO2"
optuna_storage_name="sqlite:///auto_svdiff.db"
optuna_study_name="auto_svdiff"

num_trials=10
n_repeats=50
data_size_ratio=0.01
objective_metric="max_norm_FID"
num_FID_samples=100

CUDA_VISIBLE_DEVICES=6

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_text_to_image_HPO.py --mixed_precision "fp16" --output_dir $output_dir --pretrained_model_name_or_path $pretrained_model_name_or_path --resolution $resolution --center_crop --train_batch_size $train_batch_size --gradient_checkpointing --max_train_steps $max_train_steps --learning_rate $learning_rate --max_grad_norm 1 --lr_scheduler="constant" --lr_warmup_steps 0 --unet_pretraining_type $unet_pretraining_type --num_trials $num_trials --n_repeats $n_repeats --data_size_ratio $data_size_ratio --objective_metric $objective_metric --num_FID_samples $num_FID_samples --optuna_storage_name $optuna_storage_name --optuna_study_name $optuna_study_name