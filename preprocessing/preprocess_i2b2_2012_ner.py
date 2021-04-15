# Copyright (c) 2020-present Emily Alsentzer and Facebook Inc.
# Copyright (c) 2019 Emily Alsentzer
# All rights reserved.
#
# This source code is licensed under the MIT license, which can be found here https://github.com/EmilyAlsentzer/clinicalBERT/blob/master/LICENSE
#
"""Adapted from clinicalBERT preprocessing notebooks: https://github.com/EmilyAlsentzer/clinicalBERT"""
import os, xml, xml.etree.ElementTree as ET, numpy as np, argparse

START_CDATA = "<TEXT><![CDATA["
END_CDATA = "]]></TEXT>"

TAGS = ['MEDICATION', 'OBSEE', 'SMOKER', 'HYPERTENSION', 'event', 'FAMILY_HIST']


def read_xml_file(xml_path, event_tag_type='ALL_CHILDREN', match_text=True):
    with open(xml_path, mode='r') as f:
        lines = f.readlines()
        text, in_text = [], False
        for i, l in enumerate(lines):
            if START_CDATA in l:
                text.append(list(l[l.find(START_CDATA) + len(START_CDATA):]))
                in_text = True
            elif END_CDATA in l:
                text.append(list(l[:l.find(END_CDATA)]))
                break
            elif in_text:
                text.append(list(l))

    pos_transformer = {}

    linear_pos = 1
    for line, sentence in enumerate(text):
        for char_pos, char in enumerate(sentence):
            pos_transformer[linear_pos] = (line, char_pos)
            linear_pos += 1

    try:
        xml_parsed = ET.parse(xml_path)
    except:
        with open(xml_path) as f:
            txt = f.read()
        before = txt.split('<TEXT>')[0]
        after = txt.split('</TEXT>')[1]
        with open('temporary.xml', 'w') as f:
            f.write((before + after).replace('&', 'AMPERSAND'))
        xml_parsed = ET.parse('temporary.xml')

        # raise

    tag_containers = xml_parsed.findall('TAGS')
    assert len(tag_containers) == 1, "Found multiple tag sets!"
    tag_container = tag_containers[0]

    event_tags = tag_container.findall('EVENT')
    event_labels = [['O'] * len(sentence) for sentence in text]
    for event_tag in event_tags:
        base_label = event_tag.attrib['type']
        start_pos, end_pos, event_text = event_tag.attrib['start'], event_tag.attrib['end'], event_tag.attrib['text']
        start_pos, end_pos = int(start_pos) + 1, int(end_pos)
        event_text = ' '.join(event_text.split())
        event_text = event_text.replace('AMPERSAND', '&')

        (start_line, start_char), (end_line, end_char) = pos_transformer[start_pos], pos_transformer[end_pos]

        obs_text = []
        for line in range(start_line, end_line + 1):
            t = text[line]
            s = start_char if line == start_line else 0
            e = end_char if line == end_line else len(t)
            obs_text.append(''.join(t[s:e + 1]).strip())
        obs_text = ' '.join(obs_text)
        obs_text = ' '.join(obs_text.split())

        if '&apos;' in obs_text and '&apos;' not in event_text: event_text = event_text.replace("'", "&apos;")
        if '&quot;' in obs_text and '&quot;' not in event_text: event_text = event_text.replace('"', '&quot;')

        if match_text: assert obs_text == event_text, (
                ("Texts don't match! %s v %s" % (event_text, obs_text)) + '\n' + str((
            start_pos, end_pos, line, s, e, t, xml_path
        ))
        )

        if base_label.strip() == '': continue

        event_labels[end_line][end_char] = 'I-%s' % base_label
        event_labels[start_line][start_char] = 'B-%s' % base_label

        for line in range(start_line, end_line + 1):
            t = text[line]
            s = start_char + 1 if line == start_line else 0
            e = end_char - 1 if line == end_line else len(t) - 1
            for i in range(s, e + 1): event_labels[line][i] = 'I-%s' % base_label

    return text, event_labels


def merge_into_words(text_by_char, all_labels_by_char):
    assert len(text_by_char) == len(all_labels_by_char), "Incorrect # of sentences!"

    N = len(text_by_char)

    text_by_word, all_labels_by_word = [], []

    for sentence_num in range(N):
        sentence_by_char = text_by_char[sentence_num]
        labels_by_char = all_labels_by_char[sentence_num]

        assert len(sentence_by_char) == len(labels_by_char), "Incorrect # of chars in sentence!"
        S = len(sentence_by_char)

        if labels_by_char == (['O'] * len(sentence_by_char)):
            sentence_by_word = ''.join(sentence_by_char).split()
            labels_by_word = ['O'] * len(sentence_by_word)
        else:
            sentence_by_word, labels_by_word = [], []
            text_chunks, labels_chunks = [], []
            s = 0
            for i in range(S):
                if i == S - 1:
                    text_chunks.append(sentence_by_char[s:])
                    labels_chunks.append(labels_by_char[s:])
                elif labels_by_char[i] == 'O':
                    continue
                else:
                    if i > 0 and labels_by_char[i - 1] == 'O':
                        text_chunks.append(sentence_by_char[s:i])
                        labels_chunks.append(labels_by_char[s:i])
                        s = i
                    if labels_by_char[i + 1] == 'O' or labels_by_char[i + 1][2:] != labels_by_char[i][2:]:
                        text_chunks.append(sentence_by_char[s:i + 1])
                        labels_chunks.append(labels_by_char[s:i + 1])
                        s = i + 1

            for text_chunk, labels_chunk in zip(text_chunks, labels_chunks):
                assert len(text_chunk) == len(labels_chunk), "Bad Chunking (len)"
                assert len(text_chunk) > 0, "Bad chunking (len 0)" + str(text_chunks) + str(labels_chunks)

                labels_set = set(labels_chunk)
                assert labels_set == set(['O']) or (len(labels_set) <= 3 and 'O' not in labels_set), (
                        ("Bad chunking (contents) %s" % ', '.join(labels_set)) + str(text_chunks) + str(labels_chunks)
                )

                text_chunk_by_word = ''.join(text_chunk).split()
                W = len(text_chunk_by_word)
                if W == 0:
                    continue

                if labels_chunk[0] == 'O':
                    labels_chunk_by_word = ['O'] * W
                elif W == 1:
                    labels_chunk_by_word = [labels_chunk[0]]
                elif W == 2:
                    labels_chunk_by_word = [labels_chunk[0], labels_chunk[-1]]
                else:
                    labels_chunk_by_word = [
                                               labels_chunk[0]
                                           ] + [labels_chunk[1]] * (W - 2) + [
                                               labels_chunk[-1]
                                           ]

                sentence_by_word.extend(text_chunk_by_word)
                labels_by_word.extend(labels_chunk_by_word)

        assert len(sentence_by_word) == len(labels_by_word), "Incorrect # of words in sentence!"

        if len(sentence_by_word) == 0: continue

        text_by_word.append(sentence_by_word)
        all_labels_by_word.append(labels_by_word)
    return text_by_word, all_labels_by_word


def reprocess_event_labels(folders, base_path='.', event_tag_type='event', match_text=True, dev_set_size=None):
    all_texts_by_patient, all_labels_by_patient = {}, {}

    for folder in folders:
        folder_dir = os.path.join(base_path, folder)
        xml_filenames = [x for x in os.listdir(folder_dir) if x.endswith('xml')]
        for xml_filename in xml_filenames:
            patient_num = int(xml_filename[:-4])
            xml_filepath = os.path.join(folder_dir, xml_filename)

            results = read_xml_file(
                xml_filepath,
                event_tag_type=event_tag_type,
                match_text=match_text
            )
            if results is None:
                continue
            text_by_char, labels_by_char = results
            text_by_word, labels_by_word = merge_into_words(text_by_char, labels_by_char)

            if patient_num not in all_texts_by_patient:
                all_texts_by_patient[patient_num] = []
                all_labels_by_patient[patient_num] = []

            all_texts_by_patient[patient_num].extend(text_by_word)
            all_labels_by_patient[patient_num].extend(labels_by_word)

    patients = set(all_texts_by_patient.keys())

    if dev_set_size is None:
        train_patients, dev_patients = list(patients), []
    else:
        N_train = int(len(patients) * (1 - dev_set_size))
        patients_random = np.random.permutation(list(patients))
        train_patients = list(patients_random[:N_train])
        dev_patients = list(patients_random[N_train:])

    train_texts, train_labels = [], []
    dev_texts, dev_labels = [], []

    for patient_num in train_patients:
        train_texts.extend(all_texts_by_patient[patient_num])
        train_labels.extend(all_labels_by_patient[patient_num])

    for patient_num in dev_patients:
        dev_texts.extend(all_texts_by_patient[patient_num])
        dev_labels.extend(all_labels_by_patient[patient_num])

    train_out_text_by_sentence = []
    for text, labels in zip(train_texts, train_labels):
        train_out_text_by_sentence.append('\n'.join('%s %s' % x for x in zip(text, labels)))
    dev_out_text_by_sentence = []
    for text, labels in zip(dev_texts, dev_labels):
        dev_out_text_by_sentence.append('\n'.join('%s %s' % x for x in zip(text, labels)))

    return '\n\n'.join(train_out_text_by_sentence), '\n\n'.join(dev_out_text_by_sentence)


def main(raw_data_dir, task_dir):

    final_train_text, final_dev_text = reprocess_event_labels(
        [os.path.join(raw_data_dir, '2012-07-15.original-annotation.release')], dev_set_size=0.1, match_text=True
    )

    test_text, _ = reprocess_event_labels(
        [os.path.join(raw_data_dir, '2012-08-08.test-data.event-timex-groundtruth/xml')], match_text=False, dev_set_size=None
    )

    labels = {}
    for s in final_train_text, final_dev_text, test_text:
        for line in s.split('\n'):
            if line == '': continue
            label = line.split()[-1]
            assert label == 'O' or label.startswith('B-') or label.startswith('I-'), "label wrong! %s" % label
            if label not in labels: labels[label] = 1
            else: labels[label] += 1

    with open(os.path.join(task_dir, 'train.txt.conll'), mode='w') as f:
        f.write(final_train_text)
    with open(os.path.join(task_dir, 'dev.txt.conll'), mode='w') as f:
        f.write(final_dev_text)
    with open(os.path.join(task_dir, 'test.txt.conll'), mode='w') as f:
        f.write(test_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args.raw_data_dir, args.task_dir)

