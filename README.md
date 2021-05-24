# Recipe for data augmentation based on conditional text generation

The repository contains the recipe to run data augmentation based on conditional text 
generation in order to automatically generate labeled data using a training set. 

First, we fine-tune a causal language model like GPT-2 on the training set. Each sample contains 
both 
the label and the sentence. We mark the end of the sample by a special token EOS.

Second, we synthesize labeled 
data. Given a class label, we use the fine-tuned language model to predict 
the sentence until EOS.

We show here an 
example how to 
augment GLUE SST-2 training set.


## Installation
The code runs with Python3. It is based on transformers (3.6.1) and datasets (1.6.2) libraries 
from 
HuggingFace. The script [run_clm.py](https://github.com/huggingface/transformers/tree/v4.6.
1/examples/pytorch/language-modeling/run_clm.py) from transformers examples is used for 
fine-tuning GPT2.

## Prepare the data
This following code loads and prepares the data by adding EOS token at the end of each sample. 
```shell
python prepare_data.py
```

## Fine-tuning Causal Language Model
The following code fine-tunes GPT-2 (gpt2-medium) on SST-2. The loss here is that of causal 
language modeling.

```shell
DATASET=SST-2
RAIN_FILE=$DATASET/train.txt
TEST_FILE=$DATASET/validation.txt
MODEL=gpt2-medium
MODEL_DIR=model/$MODEL-$DATASET

python3 transformers/examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path $MODEL \
    --train_file $TRAIN_FILE \
    --validation_file $TEST_FILE \
    --do_train \
    --do_eval \
    --output_dir $MODEL_DIR \
    --overwrite_output_dir
```

## Data augmentation

The following code generates samples, given the label.  
```shell
DATASET=SST-2
NUM_RETURN_SEQUENCES=800000
MODEL=gpt2-medium
MODEL_DIR=model/$MODEL-$DATASET
OUTPUT=augmented-glue-sst2/train.tsv

python data_augment.py \
 --input $DATASET/train.txt \
 --model_type gpt2 \
 --model_name_or_path $MODEL_DIR \
 --output $OUTPUT \
 --num_return_sequences $NUM_RETURN_SEQUENCES
```