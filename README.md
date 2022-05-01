# COMP9501 Machine Learning Proejct


Code for Machine Learning Project: Graphix-Tuning: A Parameter-Efficient Relational Prompt Learning Paradigm for Zero-shot Text-to-SQL (Jinyang Li & Nan Huo)




## Dependencies

To establish the environment run this code in the shell (the third line is for CUDA11.1):

``````
conda env create -f python3.7.yaml
conda activate python3.7
pip install datasets
pip install deepspeed (newest version is the stable version)
pip install torch (find your own cuda version)
``````



## Usage

### Environment setup
Activate the environment by running
``````shell
conda activate python3.7
``````

### WandB setup

Setup [WandB](https://wandb.ai/) for logging (registration needed):
``````shell
export WANDB_API_KEY=YOUR_WANDB_API_KEY
export WANDB_PROJECT=YOUR_PROJECT_NAME
export WANDB_ENTITY=YOUR_TEAM_NAME
``````

### Training



T5-3b graphix on SPIDER (4 GPUs, 32 effective batch size)
``````shell
deepspeed main.py --deepspeed deepspeed/ds_config_zero2.json --seed 2 --cfg Salesforce/T5_3b_finetune_spider.cfg --run_name T5_3b_finetune_spider --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 16 --num_train_epochs 50 --adafactor false --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/T5_3b_finetune_wikitq --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --generation_num_beams 4 --generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true
``````




