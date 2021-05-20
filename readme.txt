Recipe for transformer-based data augmentation

install transformers and datasets
4.5.1

python bin/make_job_template.py -i amr-registry.caas.intel.com/aipg/jmamou-da:latest -n clm- --run -d 1 --pwd /store/nosnap/DataAugmentDistil --command 'sh run_clm.sh' -e PYTHONIOENCODING:utf8 --pytorch

SST-2
DatasetDict({
    train: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 67349
    })
    validation: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 872
    })
    test: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 1821
    })
})

validation set
PPL gpt2-medium 37.4755
1000 33.8
2000 44.5
3000 61.2
PPL fine-tuned model 90.674

train set
PPL gpt2-medium 44.576
PPL fine-tuned model 4.45

test set
PPL gpt2-medium 37.4375
PPL fine-tuned model 135.3014




generate 800000
python bin/make_job_template.py -i amr-registry.caas.intel.com/aipg/jmamou-da:latest -n da- --run -d 2 --pwd /store/nosnap/DataAugmentDistil --command 'sh run_DA.sh' -e PYTHONIOENCODING:utf8


