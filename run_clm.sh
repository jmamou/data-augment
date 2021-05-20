export DATASET=SST-2
export TRAIN_FILE=$DATASET/train.txt
export TEST_FILE=$DATASET/validation.txt
export MODEL=gpt2-medium
#export EPOCHS=$1

python transformers/examples/language-modeling/run_clm.py \
    --model_name_or_path $MODEL \
    --train_file $TRAIN_FILE \
    --validation_file $TEST_FILE \
    --do_train \
    --do_eval \
    --output_dir model/$MODEL-$DATASET \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 --block_size 256