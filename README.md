# MemControl: Mitigating Memorization in Diffusion Models via Automated Parameter Selection
A bi-level optimisation framework that automates the selection of parameters for fine-tuning to mitigate data memorisation and ensure high-fidelity generation.  

[Publication (WACV 2025)](https://openaccess.thecvf.com/content/WACV2025/papers/Dutt_MemControl_Mitigating_Memorization_in_Diffusion_Models_via_Automated_Parameter_Selection_WACV_2025_paper.pdf)

## Preparing the Environment
- Python>=3.10.0
- Pytorch>=2.0.1+cu12.1
```
conda create -n diffusion_hpo python=3.10  
conda activate diffusion_hpo  
cd Diffusion_Memorization_HPO  
pip install -r requirements.txt
```


## Stage 1: Run the HPO to Search for the Best Mask (Parameter Subset)
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