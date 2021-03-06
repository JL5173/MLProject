python train.py --seed 2 --cfg Salesforce/T5_base_finetune_wikitq.cfg --run_name T5_base_finetune_wikitq \
--logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps \
--eval_steps 5 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 \
--save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 \
--adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate \
--output_dir output/T5_base_finetune_wikitq --per_device_train_batch_size 4 \
--per_device_eval_batch_size 16 --generation_num_beams 4 --generation_max_length 128 --input_max_length 1024 \
--ddp_find_unused_parameters true --overwrite_output_dir true

python train.py --seed 2 --cfg Salesforce/T5_base_prefix_wikitq.cfg --run_name T5_base_prefix_wikitq \
--logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 5 \
--metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 1 \
--load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 --adafactor true --learning_rate 5e-5 \
--do_train --do_eval --do_predict --predict_with_generate --output_dir output/T5_base_prefix_wikitq --overwrite_output_dir \
--per_device_train_batch_size 4 --per_device_eval_batch_size 16 --generation_num_beams 4 --generation_max_length 128 \
--input_max_length 1024 --ddp_find_unused_parameters true --overwrite_output_dir true