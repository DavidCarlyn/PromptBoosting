python ensemble_training_proposal.py \
    --dataset mnli --classifier xgboost \
    --model roberta --label_set_size 3 --change_template \
    --use_part_templates --start_idx 0 --end_idx 10 \
    --sort_dataset --fewshot --fewshot_k 16 --fewshot_seed 13