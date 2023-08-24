# Tal Fiskus (ID: 208423707)
# Yoel Benabou (ID: 342875648)

# utils class
import os

import numpy as np
import time
import matplotlib.pyplot as plt
import scipy


def extract_data_from_file_to_array(seq_file_path):
    """
    Extract data from input files to array.
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
    Extract the data from the input files.
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
        # get first 1 million lines
        for index, line in enumerate(lines):
            if index == 1000000:  # 1 million
                break
            if '\t' in line:
                extracted_data.append(line.strip().split('\t')[0])
    # Calculate and print total runtime
    print("Total runtime of extract_data_from_rbns_file:", (time.time() - start_time), "seconds")
    return np.array(extracted_data)


def create_train_and_test_set_from_seq_files(seq_files_paths_list, one_hot_encoded_size, sub_sequence_length):
    """
    Create the train and test set from the received files.
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
    One hot encoding for a RNA sequence.
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
    Convert list of RNA sequences.
    :param sequences:
    :param one_hot_encoded_size:
    :param sub_sequence_length:
    :return:
    """
    seq = np.zeros((len(sequences), sub_sequence_length, one_hot_encoded_size), dtype=np.float32)
    for index, sequence in np.ndenumerate(sequences):
        if len(sequence) > sub_sequence_length:
            new_sequence = sequence[0:sub_sequence_length]
            seq[index] = one_hote(new_sequence)
        else:
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


def write_predictions_to_file(predictions):
    """
    Write the predictions list into a txt file
    :param predictions:
    :param protein_number:
    :return:
    """
    file_path = "Prediction_208423707_342875648.txt"

    with open(file_path, 'w') as file:
        for value in predictions:
            file.write(f"{value}\n")


def generate_graph(data):
    """
    Generate the Pearson correlation graph
    :param data:
    :return:
    """
    names = list(data.keys())
    values = [corr[0] for corr in data.values()]

    mean_value = sum(values) / len(values)

    # Create a bar graph
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.bar(names, values)

    plt.axhline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')

    # Customize the plot (optional)
    plt.xlabel('Protein number')
    plt.ylabel('Pearson Correlation')
    plt.title('Pearson Correlation between real values and RBNS training predictions')
    plt.legend()  # Show legend with the updated label
    plt.tight_layout()
    plt.savefig("correlation_graph.png")
    plt.show()


def calculate_and_graph_pearson_for_all():
    """
    Calculate the pearson correlation for all the proteins predicitons files
    :return:
    """
    directory = 'Predictions'
    data = {}

    # Iterate over files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            # Extract the key (name without '.txt' extension)
            key = filename[:-4]  # Remove the last 4 characters (.txt)

            real_rna_values_file = "RNCMPT_training/RBP" + key + ".txt"
            real_rna_values = np.array(extract_data_from_file_to_array(real_rna_values_file),
                                       dtype=np.float32)

            rna_predictions_file = "Predictions/" + filename
            rna_predictions_values = np.array(extract_data_from_file_to_array(rna_predictions_file),
                                              dtype=np.float32)

            correlation = scipy.stats.pearsonr(rna_predictions_values, real_rna_values)

            # Add the key-value pair to the dictionary
            data[key] = correlation

            print("Pearson correlation for protein " + str(key) + " = ", correlation)

    generate_graph(data)
