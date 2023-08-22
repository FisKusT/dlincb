# Project in Deep Learning in Computational Biology course.

import scipy
import dlincb_utils
import numpy as np
from nn_model import RBNS_Classifier
import time


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sub_sequence_length = 20
    one_hot_encoded_size = 4

    # Measure start time
    start_time = time.time()

    # Parse RBNS data for protein
    seq_file_list = [
        "RBNS_training/RBP1_input.seq",
        "RBNS_training/RBP1_1300nM.seq"
    ]

    x_one_hot, y_labels = dlincb_utils.create_train_and_test_set_from_seq_files(seq_file_list,
                                                                                one_hot_encoded_size=one_hot_encoded_size,
                                                                                sub_sequence_length=sub_sequence_length)

    # make and train model
    start_time2 = time.time()

    rbns_classifier = RBNS_Classifier(use_load_model=True, one_hot_encoded_size=one_hot_encoded_size,
                                      sub_sequence_length=sub_sequence_length)
    rbns_classifier.train_model(x_one_hot, y_labels)

    print("Total runtime of training data:", (time.time() - start_time2), "seconds")

    # Parse RNA data
    start_time3 = time.time()

    rna_file = "RNAcompete_sequences.txt"
    rna_sequences = dlincb_utils.extract_data_from_file_to_array(rna_file)
    all_sequences, max_combinaisons = dlincb_utils.generate_subsequences(rna_sequences,
                                                                         sub_sequence_length=sub_sequence_length)

    print("Total runtime of generate subsequences data:", (time.time() - start_time3), "seconds")

    start_time4 = time.time()

    rna_predictions = rbns_classifier.do_prediction(all_sequences, max_combinaisons, one_hot_encoded_size=one_hot_encoded_size,
                                                    sub_sequence_length=sub_sequence_length)
    print("Total runtime of predict data:", (time.time() - start_time4), "seconds")

    print("Total runtime:", (time.time() - start_time), "seconds")

    # get the real value predictions
    real_rna_values_file = "C:/Users/sorte/Desktop/BIU/DL in CB/project/RNCMPT_training/RBP1.txt"
    real_rna_values = np.array(dlincb_utils.extract_data_from_file_to_array(real_rna_values_file),
                               dtype=np.float32)


    # calculate pearson correlation for
    correlation1 = scipy.stats.pearsonr(real_rna_values, rna_predictions)
    # correlation2 = scipy.stats.pearsonr(real_rna_values, rna_predictions2)
    print("Pearson correlation1", correlation1)
    # print("Pearson correlation2", correlation2)

    # Calculate and print total runtime
    print("Total runtime of a single protein:", (time.time() - start_time), "seconds")

    print("END!")
