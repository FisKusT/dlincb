# utils class

import numpy as np
import time


def extract_data_from_file_to_array(seq_file_path):
    """

    :param seq_file_path:
    :return:
    """
    extracted_data = []
    with open(seq_file_path, 'r') as seq_file:
        for line in seq_file:
            extracted_data.append(line.strip())
    return np.array(extracted_data)


def extract_data_from_rbns_file(seq_file_path):
    """

    :param seq_file_path:
    :return:
    """
    # Measure start time
    start_time = time.time()
    extracted_data = []
    with open(seq_file_path, 'r') as seq_file:
        lines = seq_file.readlines()
        # Shuffle the lines
        # random.shuffle(lines)
        # get first 500K lines
        for index, line in enumerate(lines):
            if index == 500000:
                break
            if '\t' in line:
                extracted_data.append(line.strip().split('\t')[0])
    # Calculate and print total runtime
    print("Total runtime of extract_data_from_rbns_file:", (time.time() - start_time), "seconds")
    return np.array(extracted_data)


def create_train_and_test_set_from_seq_files(seq_files_paths_list, one_hot_encoded_size, sub_sequence_length):
    """

    :param seq_files_paths_list:
    :param one_hot_encoded_size:
    :param sub_sequence_length:
    :return:
    """
    # Measure start time
    start_time = time.time()
    sequences = np.empty(0)
    y_labels = np.empty(0)
    for index, seq_file_path in enumerate(seq_files_paths_list):
        data = extract_data_from_rbns_file(seq_file_path)
        labels = np.full(len(data), index)
        sequences = np.append(sequences, data)
        y_labels = np.append(y_labels, labels)
    x_data = convert_all_to_one_hot(sequences, one_hot_encoded_size=one_hot_encoded_size,
                                    sub_sequence_length=sub_sequence_length)
    # Measure end time
    print("Total runtime of create_train_and_test_set_from_seq_files:", (time.time() - start_time), "seconds")
    return x_data, y_labels


def one_hot_encoding_rna_sequences_by_batch(batch_sequences, max_combinaisons, one_hot_encoded_size,
                                            sub_sequence_length):
    """
    One hot encoding of each subsequences and adding padding to have all sequences with same number of subsequences
    :param one_hot_encoded_size:
    :param sub_sequence_length:
    :param batch_sequences:
    :param max_combinaisons:
    :return:
    """

    one_hot_batch = np.zeros((len(batch_sequences), max_combinaisons, sub_sequence_length, one_hot_encoded_size),
                             dtype=np.float32)

    for i, original_sequence in enumerate(batch_sequences):
        for j, subsequence in enumerate(original_sequence):
            for k, nucleotide in enumerate(subsequence):
                if nucleotide == 'A':
                    one_hot_batch[i, j, k, 0] = 1.0
                elif nucleotide == 'C':
                    one_hot_batch[i, j, k, 1] = 1.0
                elif nucleotide == 'G':
                    one_hot_batch[i, j, k, 2] = 1.0
                elif nucleotide == 'U':
                    one_hot_batch[i, j, k, 3] = 1.0
                else:
                    one_hot_batch[i, j, k, :] = 0.25

    return one_hot_batch


def one_hote(sequence):
    """

    :param sequence:
    :return:
    """
    mapping = {"A": [1.0, 0.0, 0.0, 0.0],
               "C": [0.0, 1.0, 0.0, 0.0],
               "G": [0.0, 0.0, 1.0, 0.0],
               "T": [0.0, 0.0, 0.0, 1.0],
               "U": [0.0, 0.0, 0.0, 1.0],
               "N": [0.25, 0.25, 0.25, 0.25]}
    seq2 = [mapping[i] for i in sequence]
    return seq2


def convert_all_to_one_hot(sequences, one_hot_encoded_size, sub_sequence_length):
    """

    :param sequences:
    :param one_hot_encoded_size:
    :param sub_sequence_length:
    :return:
    """
    seq = np.zeros((len(sequences), sub_sequence_length, one_hot_encoded_size), dtype=np.float32)
    for index, sequence in np.ndenumerate(sequences):
        seq[index] = one_hote(sequence)

    return seq


def generate_subsequences(sequences, sub_sequence_length):
    """
    Generate all the subsequences of size l for each RNA sequence.
    :param sub_sequence_length: the length of the subsequence we want to generate
    :param sequences: All the sequences we extracted from the file
    :return: the subsequences and the maximum number of subsequences
    """
    subsequences_mapping = []
    max_subsequences = 0

    for sequence in sequences:
        sequence_length = len(sequence)

        if sequence_length == sub_sequence_length:
            subsequences_mapping.append(sequence)
        else:
            sequence_subsequences = []
            for i in range(sequence_length - (sub_sequence_length - 1)):
                subsequence = sequence[i:i + sub_sequence_length]
                sequence_subsequences.append(subsequence)

            if len(sequence_subsequences) > max_subsequences:
                max_subsequences = len(sequence_subsequences)

            subsequences_mapping.append(sequence_subsequences)

    return subsequences_mapping, max_subsequences


def clean_padding_from_prediction(data, predictions, batch_size, max_combinaisons):
    """
    Put -inf if the prediction was for the padding to not take into consideration the padding prediciton.
    :return: The max prediciton for each sequence in the batch
    """
    max_preds = []
    for batch_sequences, batch_predictions in zip(data, predictions):
        for sequence, prediction in zip(batch_sequences, batch_predictions):
            if np.sum(sequence) > 0:
                max_preds.append((prediction * 3) - 1)
            else:
                max_preds.append(float('-inf'))

    return np.array(max_preds).reshape(batch_size, max_combinaisons)
