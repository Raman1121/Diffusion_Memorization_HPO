# Diffusion_Memorization_HPO
A framework to reduce memorization in text-to-image diffusion models using HPO


## Stage 1: Run the HPO
```
python train_text_to_image_HPO.py --mixed_precision "fp16" --output_dir <output_dir> \
                                            --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" --resolution 224 \
                                            --center_crop --train_batch_size <train_batch_size> --gradient_checkpointing \
                                            --max_train_steps <max_train_steps> --max_grad_norm 1 \
                                            --lr_scheduler="constant" --lr_warmup_steps 0 \
                                            --unet_pretraining_type <PEFT_METHOD (auto_svdiff or auto_difffit or auto_attention)> \
                                            --num_trials <NUM_HPO_TRIALS> \
                                            --n_repeats <NUM_REPEATS> --data_size_ratio 0.01 --objective_metric "max_norm_FID" \
                                            --num_FID_samples 1000 --optuna_storage_name \
                                            <optuna_storage_name>  --optuna_study_name <optuna_study_name>
```

## Stage 2: Fine-tune with the Best Mask
```
python train_text_to_image.py --mixed_precision "fp16" --output_dir $output_dir \
                                            --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" --resolution 224 \
                                            --center_crop --train_batch_size <<train_batch_size>> --gradient_checkpointing \
                                            --max_train_steps <max_train_steps> --learning_rate <learning_rate> \
                                            --max_grad_norm 1 --lr_scheduler="constant" --lr_warmup_steps 0 \
                                            --unet_pretraining_type <PEFT_METHOD> \
```                                            