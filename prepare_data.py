from datasets import load_dataset

EOS = '</s>'

for split in {'train', 'validation', 'test'}:
    print(split)
    dataset = load_dataset('glue', 'sst2', split=split)
    print(dataset)
    with open(split+'.txt', 'w') as fw:
        for d in dataset:
            fw.write(d['sentence'] + EOS + '\n')