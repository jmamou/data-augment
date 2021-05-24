from datasets import load_dataset

from utils import EOS

for split in {'train', 'validation'}:
    dataset = load_dataset('glue', 'sst2', split=split)
    print(dataset)
    with open('SST-2/'+split+'.txt', 'w') as fw:
        for d in dataset:
            fw.write(str(d['label'])+ '\t' +d['sentence'] + EOS + '\n')