DATASET=SST-2
NUM_RETURN_SEQUENCES=100
MODEL=gpt2-medium
MODEL_DIR=model/$MODEL-$DATASET
OUTPUT=augmented-glue-sst2/train.tsv

python data_augment.py \
 --input $DATASET/train.txt \
 --model_type gpt2 \
 --model_name_or_path $MODEL_DIR \
 --output $OUTPUT \
 --num_return_sequences $NUM_RETURN_SEQUENCES