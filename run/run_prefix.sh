python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234 train.py --seed 2 \
--cfg Salesforce/T5_base_prefix_wikitq.cfg --run_name T5_base_prefix_wikitq --logging_strategy steps \
--logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 \
--metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 \
--save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 \
--adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate \
--output_dir output/T5_base_prefix_wikitq --overwrite_output_dir --per_device_train_batch_size 4 \
--per_device_eval_batch_size 16 --generation_num_beams 4 --generation_max_length 128 --input_max_length 1024 \
--ddp_find_unused_parameters true

'''W / O distributed'''

python train.py --seed 1 --cfg Salesforce/T5_base_prefix_spider_with_cell_value.cfg --run_name T5_base_prefix10_spider_Patrick \
--logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 \
--metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 \
--save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 \
--adafactor true --learning_rate 5e-5 --do_train true --do_eval true --predict_with_generate true \
--output_dir output/T5_base_prefix10_spider --per_device_train_batch_size 4 --per_device_eval_batch_size 32 \
--generation_num_beams 1 --generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true \
--report_to wandb --overwrite_output_dir true

'''Prefix30-Base'''

python train.py --seed 1 --cfg Salesforce/T5_base_prefix_spider_with_cell_value.cfg --run_name T5_base_prefix30_spider \
--logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 \
--metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 \
--save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 \
--adafactor true --learning_rate 5e-5 --do_train true --do_eval true --predict_with_generate true \
--output_dir output/T5_base_prefix30_spider --per_device_train_batch_size 4 --per_device_eval_batch_size 32 \
--generation_num_beams 1 --generation_max_length 128 --input_max_length 512 --ddp_find_unused_parameters true \
--report_to wandb --overwrite_output_dir true

'''Prefix10-Large'''

python train.py --seed 1 --cfg Salesforce/T5_large_prefix_spider_with_cell_value.cfg --run_name T5_large_prefix10_spider_Patrick \
--logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 \
--metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 \
--save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 \
--adafactor true --learning_rate 5e-5 --do_train true --do_eval true --predict_with_generate true \
--output_dir output/T5_large_prefix10_spider --per_device_train_batch_size 4 --per_device_eval_batch_size 32 \
--generation_num_beams 1 --generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true \
--report_to wandb --overwrite_output_dir true

'''Prefix30-Large'''

python train.py --seed 1 --cfg Salesforce/T5_large_prefix_spider_with_cell_value.cfg --run_name T5_large_prefix30_spider \
--logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 \
--metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 \
--save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 \
--adafactor true --learning_rate 5e-5 --do_train true --do_eval true --predict_with_generate true \
--output_dir output/T5_large_prefix30_spider --per_device_train_batch_size 4 --per_device_eval_batch_size 32 \
--generation_num_beams 1 --generation_max_length 128 --input_max_length 512 --ddp_find_unused_parameters true \
--report_to wandb --overwrite_output_dir true

'''Prefix50-Large'''

python train.py --seed 1 --cfg Salesforce/T5_large_prefix_spider_with_cell_value.cfg --run_name T5_large_prefix50_spider \
--logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 \
--metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 \
--save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 \
--adafactor true --learning_rate 5e-5 --do_train true --do_eval true --predict_with_generate true \
--output_dir output/T5_large_prefix50_spider --per_device_train_batch_size 4 --per_device_eval_batch_size 32 \
--generation_num_beams 1 --generation_max_length 128 --input_max_length 512 --ddp_find_unused_parameters true \
--report_to wandb --overwrite_output_dir true

'''Prefix50-Base'''

python train.py --seed 1 --cfg Salesforce/T5_base_prefix_spider_with_cell_value.cfg --run_name T5_base_prefix50_spider \
--logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 \
--metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 \
--save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 \
--adafactor true --learning_rate 5e-5 --do_train true --do_eval true --predict_with_generate true \
--output_dir output/T5_base_prefix50_spider --per_device_train_batch_size 4 --per_device_eval_batch_size 32 \
--generation_num_beams 1 --generation_max_length 128 --input_max_length 512 --ddp_find_unused_parameters true \
--report_to wandb --overwrite_output_dir true

'''Prefix70-Base'''
python train.py --seed 1 --cfg Salesforce/T5_base_prefix_spider_with_cell_value.cfg --run_name T5_base_prefix70_spider \
--logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 \
--metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 \
--save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 \
--adafactor true --learning_rate 5e-5 --do_train true --do_eval true --predict_with_generate true \
--output_dir output/T5_base_prefix70_spider --per_device_train_batch_size 4 --per_device_eval_batch_size 32 \
--generation_num_beams 1 --generation_max_length 128 --input_max_length 512 --ddp_find_unused_parameters true \
--report_to wandb --overwrite_output_dir true

'''Prefix70-Large'''
python train.py --seed 1 --cfg Salesforce/T5_large_prefix_spider_with_cell_value.cfg --run_name T5_large_prefix70_spider \
--logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 \
--metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 \
--save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 \
--adafactor true --learning_rate 5e-5 --do_train true --do_eval true --predict_with_generate true \
--output_dir output/T5_large_prefix70_spider --per_device_train_batch_size 4 --per_device_eval_batch_size 32 \
--generation_num_beams 1 --generation_max_length 128 --input_max_length 512 --ddp_find_unused_parameters true \
--report_to wandb --overwrite_output_dir true

'''Prefix90-Base'''

python train.py --seed 1 --cfg Salesforce/T5_base_prefix_spider_with_cell_value.cfg --run_name T5_base_prefix90_spider \
--logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 \
--metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 \
--save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 \
--adafactor true --learning_rate 5e-5 --do_train true --do_eval true --predict_with_generate true \
--output_dir output/T5_base_prefix90_spider --per_device_train_batch_size 4 --per_device_eval_batch_size 32 \
--generation_num_beams 1 --generation_max_length 128 --input_max_length 512 --ddp_find_unused_parameters true \
--report_to wandb --overwrite_output_dir true


export WANDB_API_KEY=46555a76c0563ad8fcea6f7edc073d17f1fd63cc
export WANDB_PROJECT=text-to-sql
export WANDB_ENTITY=YOUR_TEAM_NAME