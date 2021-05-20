#!/usr/bin/env python3
# coding=utf-8

# adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-generation/run_generation.py

import argparse
import logging
import random
from collections import defaultdict

import numpy as np
import torch

from tqdm import tqdm

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer, pipeline,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    """# Data augmentation based on conditional text generation using the auto-regressive models of the library: GPT, GPT-2, Transformer-XL, XLNet, CTRL."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_prefix", type=int, default=1, help="length of the prefix "
                                                                   "to sample from the input file for the conditional text generation")
    parser.add_argument('--input', type=str, help="input file to augment by sampling prefixes.")
    parser.add_argument('--output', type=str, help="output file to store augmented data")
    parser.add_argument("--model_type", default='gpt2', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
                        )
    parser.add_argument("--model_name_or_path", default='gpt2', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            MODEL_CLASSES.keys()),
                        )
    parser.add_argument("--stop_token", type=str, default='</s>',
                        help="Token at which text generation is stopped")

    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
                        )
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2"
                        )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--xlm_language", type=str, default="",
                        help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="The number of samples to generate.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
                        )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError(
            "the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    if args.fp16:
        model.half()

    # initialize text generation pipeline
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer,
                              device=0)

    logger.info(args)

    # map between the prefix and its number of occurrences in the input file
    prefix2count = defaultdict(int)
    with open(args.input, 'r', encoding='utf8') as file_in:
        for sentence in file_in:
            sentence = sentence.split()
            if len(sentence) > args.seed_prefix:
                text_input = ' '.join(sentence[0:args.seed_prefix])
                prefix2count[text_input] += 1

    total_count = sum(prefix2count.values())
    factor = args.num_return_sequences / total_count + 1
    prefixes = list(prefix2count.keys())
    random.shuffle(prefixes)

    with open(args.output, 'w', encoding='utf8') as file:
        for prefix in tqdm(prefixes):
            count = prefix2count[prefix]
            output_sequences = text_generator(text_inputs=prefix + ' ', early_stopping=True,
                                              temperature=args.temperature,
                                              top_k=args.k,
                                              top_p=args.p,
                                              repetition_penalty=args.repetition_penalty,
                                              do_sample=True,
                                              num_return_sequences=int(count * factor),
                                              clean_up_tokenization_spaces=True,
                                              return_full_text=True)
            augmented = set()
            for seq in output_sequences:
                text = seq['generated_text']
                text = text[:text.find(args.stop_token) if args.stop_token and text.find(
                    args.stop_token) > -1 else None].strip()
                if text is not None and text not in augmented:
                    file.write(text + '\n')
                    augmented.add(text)
            file.flush()


if __name__ == "__main__":
    main()
