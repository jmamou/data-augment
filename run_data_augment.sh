DATASET=SST-2
NUM_RETURN_SEQUENCES=800000
#MODEL=model/gpt2-medium-SST-2-3
MODEL=gpt2
SEED_PREFIX=1

python run_data_augment.py --seed_prefix $SEED_PREFIX \
 --input $DATASET/train.txt \
 --stop_token '</s>' \
 --model_type gpt2 \
 --model_name_or_path $MODEL \
 --csv $DATASET/DA-gpt2-medium-SST-2-3.csv \
 --num_return_sequences $NUM_RETURN_SEQUENCES --no_cuda