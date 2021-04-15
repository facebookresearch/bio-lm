# Copyright (c) 2020-present Emily Alsentzer and Facebook Inc.
# Copyright (c) 2019 Emily Alsentzer
# All rights reserved.
#
# This source code is licensed under the MIT license, which can be found here https://github.com/EmilyAlsentzer/clinicalBERT/blob/master/LICENSE
#
"""Adapted from clinicalBERT preprocessing notebooks: https://github.com/EmilyAlsentzer/clinicalBERT"""
import os, re, pickle, numpy as np
import argparse


def process_concept(concept_str):
    """
    takes string like
    'c="asymptomatic" 16:2 16:2||t="problem"'
    and returns dictionary like
    {'t': 'problem', 'start_line': 16, 'start_pos': 2, 'end_line': 16, 'end_pos': 2}
    """
    try:
        position_bit, problem_bit = concept_str.split('||')
        t = problem_bit[3:-1]

        start_and_end_span = next(re.finditer('\s\d+:\d+\s\d+:\d+', concept_str)).span()
        c = concept_str[3:start_and_end_span[0]-1]
        c = [y for y in c.split(' ') if y.strip() != '']
        c = ' '.join(c)

        start_and_end = concept_str[start_and_end_span[0]+1 : start_and_end_span[1]]
        start, end = start_and_end.split(' ')
        start_line, start_pos = [int(x) for x in start.split(':')]
        end_line, end_pos = [int(x) for x in end.split(':')]

    except:
        raise

    return {
        't': t, 'start_line': start_line, 'start_pos': start_pos, 'end_line': end_line, 'end_pos': end_pos,
        'c': c,
    }


def build_label_vocab(base_dirs):
    seen, label_vocab, label_vocab_size = set(['O']), {'O': 'O'}, 0

    for base_dir in base_dirs:
        concept_dir = os.path.join(base_dir, 'concept')

        assert os.path.isdir(concept_dir), "Directory structure doesn't match!"

        ids = set([x[:-4] for x in os.listdir(concept_dir) if x.endswith('.con')])

        for i in ids:
            with open(os.path.join(concept_dir, '%s.con' % i)) as f:
                concepts = [process_concept(x.strip()) for x in f.readlines()]
            for c in concepts:
                if c['t'] not in seen:
                    label_vocab_size += 1
                    label_vocab['B-%s' % c['t']] = 'B-%s' % c['t'] # label_vocab_size
                    label_vocab_size += 1
                    label_vocab['I-%s' % c['t']] = 'I-%s' % c['t'] # label_vocab_size
                    seen.update([c['t']])
    return label_vocab, label_vocab_size


def reformatter(base, label_vocab, txt_dir = None, concept_dir = None):
    if txt_dir is None: txt_dir = os.path.join(base, 'txt')
    if concept_dir is None: concept_dir = os.path.join(base, 'concept')
    assert os.path.isdir(txt_dir) and os.path.isdir(concept_dir), "Directory structure doesn't match!"

    txt_ids = set([x[:-4] for x in os.listdir(txt_dir) if x.endswith('.txt')])
    concept_ids = set([x[:-4] for x in os.listdir(concept_dir) if x.endswith('.con')])

    assert txt_ids == concept_ids, (
        "id set doesn't match: txt - concept = %s, concept - txt = %s"
        "" % (str(txt_ids - concept_ids), str(concept_ids - txt_ids))
    )

    ids = txt_ids

    reprocessed_texts = {}
    for i in ids:
        with open(os.path.join(txt_dir, '%s.txt' % i), mode='r') as f:
            lines = f.readlines()
            txt = [[y for y in x.strip().split(' ') if y.strip() != ''] for x in lines]
            line_starts_with_space = [x.startswith(' ') for x in lines]
        with open(os.path.join(concept_dir, '%s.con' % i), mode='r') as f:
            concepts = [process_concept(x.strip()) for x in f.readlines()]

        labels = [['O' for _ in line] for line in txt]
        for c in concepts:
            if c['start_line'] == c['end_line']:
                line = c['start_line']-1
                p_modifier = -1 if line_starts_with_space[line] else 0
                text = (' '.join(txt[line][c['start_pos']+p_modifier:c['end_pos']+1+p_modifier])).lower()
                assert text == c['c'], (
                    "Text mismatch! %s vs. %s (id: %s, line: %d)\nFull line: %s"
                    "" % (c['c'], text, i, line, txt[line])
                )

            for line in range(c['start_line']-1, c['end_line']):
                p_modifier = -1 if line_starts_with_space[line] else 0
                start_pos = c['start_pos']+p_modifier if line == c['start_line']-1 else 0
                end_pos   = c['end_pos']+1+p_modifier if line == c['end_line']-1 else len(txt[line])

                if line == c['end_line'] - 1: labels[line][end_pos-1] = label_vocab['I-%s' % c['t']]
                if line == c['start_line'] - 1: labels[line][start_pos] = label_vocab['B-%s' % c['t']]
                for j in range(start_pos + 1, end_pos-1): labels[line][j] = label_vocab['I-%s' % c['t']]

        joined_words_and_labels = [zip(txt_line, label_line) for txt_line, label_line in zip(txt, labels)]

        out_str = '\n\n'.join(
            ['\n'.join(['%s %s' % p for p in joined_line]) for joined_line in joined_words_and_labels]
        )

        reprocessed_texts[i] = out_str

    return reprocessed_texts


def main(beth_dir, partners_dir, test_dir, test_txt_dir, task_dir):
    label_vocab, label_vocab_size = build_label_vocab([beth_dir, partners_dir])

    reprocessed_texts = {
        'beth':     reformatter(beth_dir, label_vocab),
        'partners': reformatter(partners_dir, label_vocab),
        'test':     reformatter(
            test_dir, label_vocab,
            txt_dir=test_txt_dir,
            concept_dir=os.path.join(test_dir, 'concepts')
        ),
    }
    np.random.seed(1)
    all_partners_train_ids = np.random.permutation(list(reprocessed_texts['partners'].keys()))
    N = len(all_partners_train_ids)
    N_train = int(0.9 * N)

    partners_train_ids = all_partners_train_ids[:N_train]
    partners_dev_ids = all_partners_train_ids[N_train:]
    print("Partners # Patients: Train: %d, Dev: %d" %(len(partners_train_ids), len(partners_dev_ids)))
    all_beth_train_ids = np.random.permutation(list(reprocessed_texts['beth'].keys()))
    N = len(all_beth_train_ids)
    N_train = int(0.9 * N)

    beth_train_ids = all_beth_train_ids[:N_train]
    beth_dev_ids = all_beth_train_ids[N_train:]
    print("Beth # Patients: Train: %d, Dev: %d" % (len(beth_train_ids), len(beth_dev_ids)))

    print("Merged # Patients: Train: %d, Dev: %d" % (
      len(partners_train_ids) + len(beth_train_ids), len(beth_dev_ids) + len(partners_dev_ids)
    ))

    merged_train_txt = '\n\n'.join(np.random.permutation(
        [reprocessed_texts['partners'][i] for i in partners_train_ids] +
        [reprocessed_texts['beth'][i] for i in beth_train_ids]
    ))
    merged_dev_txt = '\n\n'.join(np.random.permutation(
        [reprocessed_texts['partners'][i] for i in partners_dev_ids] +
        [reprocessed_texts['beth'][i] for i in beth_dev_ids]
    ))
    merged_test_txt = '\n\n'.join(np.random.permutation(list(reprocessed_texts['test'].values())))

    print("Merged # Samples: Train: %d, Dev: %d, Test: %d" % (
        len(merged_train_txt.split('\n\n')),
        len(merged_dev_txt.split('\n\n')),
        len(merged_test_txt.split('\n\n'))
    ))

    partners_train_txt = '\n\n'.join(np.random.permutation(
        [reprocessed_texts['partners'][i] for i in partners_train_ids]
    ))
    partners_dev_txt = '\n\n'.join(np.random.permutation(
        [reprocessed_texts['partners'][i] for i in partners_dev_ids]
    ))
    partners_test_txt = '\n\n'.join(np.random.permutation(list(reprocessed_texts['test'].values())))

    OUT_FILES = {
        'merged_train': os.path.join(task_dir, 'merged', 'train.tsv'),
        'merged_dev':  os.path.join(task_dir, 'merged', 'dev.tsv'),
        'merged_test':  os.path.join(task_dir, 'merged', 'test.tsv'),
        'partners_train': os.path.join(task_dir, 'merged', 'train.tsv'),
        'partners_dev':  os.path.join(task_dir, 'merged', 'dev.tsv'),
        'partners_test':  os.path.join(task_dir, 'merged', 'test.tsv'),
        'vocab': os.path.join(task_dir, 'merged' 'labels.txt')
    }
    os.makedirs(os.path.join(task_dir, 'merged'), exist_ok=True)
    os.makedirs(os.path.join(task_dir, 'partners'), exist_ok=True)

    with open(OUT_FILES['merged_train'], mode='w') as f: f.write(merged_train_txt)
    with open(OUT_FILES['merged_dev'], mode='w') as f: f.write(merged_dev_txt)
    with open(OUT_FILES['merged_test'], mode='w') as f: f.write(merged_test_txt)
    with open(OUT_FILES['partners_train'], mode='w') as f: f.write(partners_train_txt)
    with open(OUT_FILES['partners_dev'], mode='w') as f: f.write(partners_dev_txt)
    with open(OUT_FILES['partners_test'], mode='w') as f: f.write(partners_test_txt)
    with open(OUT_FILES['vocab'], mode='w') as f: f.write('\n'.join(label_vocab.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beth_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--partners_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test_txt_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    # beth_dir = './data/blue_raw_data/data/i2b2-2010/original/concept_assertion_relation_training_data/beth/'
    # partners_dir = './data/blue_raw_data/data/i2b2-2010/original/concept_assertion_relation_training_data/partners/'
    # test_dir = './data/blue_raw_data/data/i2b2-2010/original/reference_standard_for_test_data/'
    # test_txt_dir = './data/blue_raw_data/data/i2b2-2010/original/test_data/'
    # task_dir = 'data/I2B22010NER'
    main(args.beth_dir, args.partners_dir, args.test_dir, args.test_txt_dir, args.task_dir)
