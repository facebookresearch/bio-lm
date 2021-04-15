# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm


def main(args):
    """Inputs and cleans and ensures no inputs are too long
    Adapted from https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py
    """

    subword_len_counter = 0
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    for line in tqdm(open(args.filename), desc=f'processing {args.filename}'):
        line = line.rstrip()

        if not line:
            print(line)
            subword_len_counter = 0
            continue

        token = line.split()[0]
        current_subwords_len = len(tokenizer.tokenize(token))

        # Token contains strange control characters like \x96 or \x95
        # Just filter out the complete line
        if current_subwords_len == 0:
            continue

        if (subword_len_counter + current_subwords_len) > args.max_len:
            print("")
            print(line)
            subword_len_counter = 0
            continue

        subword_len_counter += current_subwords_len

        print(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='roberta-large',
    )
    parser.add_argument(
        "--max_len",
        default=512,
        type=int,
    )
    args = parser.parse_args()
    main(args)
