# utils class

import numpy as np
import time


def extract_data_from_file_to_array(seq_file_path):
    extracted_data = []
    with open(seq_file_path, 'r') as seq_file:
        for line in seq_file:
            extracted_data.append(line.strip())
    return np.array(extracted_data)


def extract_data_from_rbns_file(seq_file_path):
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


def create_train_and_test_set_from_seq_files(seq_files_paths_list):
    # Measure start time
    start_time = time.time()
    sequences = np.empty(0)
    y_labels = np.empty(0)
    for index, seq_file_path in enumerate(seq_files_paths_list):
        data = extract_data_from_rbns_file(seq_file_path)
        labels = np.full(len(data), index)
        print(len(data))
        sequences = np.append(sequences, data)
        y_labels = np.append(y_labels, labels)
    x_data = convert_sequences_to_one_hot_vector(sequences)
    # Measure end time
    print("Total runtime of create_train_and_test_set_from_seq_files:", (time.time() - start_time), "seconds")
    return x_data, y_labels


def convert_sequences_to_one_hot_vector(sequences):
    # Create a mapping of nucleotides to indices
    nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # Convert sequences to one-hot encoded tensor
    one_hot_tensor = np.zeros((len(sequences), 20, 4), dtype=np.float32)

    for i, sequence in enumerate(sequences):
        for j, nucleotide in enumerate(sequence):
            if nucleotide == 'N':
                one_hot_tensor[i, j] = 0.25
            else:
                nucleotide_index = nucleotide_to_index.get(nucleotide, None)
                if nucleotide_index is not None:
                    one_hot_tensor[i, j, nucleotide_index] = 1.0

    return one_hot_tensor
