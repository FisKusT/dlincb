# Project in Deep Learning in Computational Biology course.

import dlincb_utils
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
        "RBNS_training/RBP8_input.seq",
        "RBNS_training/RBP8_1300nM.seq"
    ]

    x_one_hot, y_labels = dlincb_utils.create_train_and_test_set_from_seq_files(seq_file_list,
                                                                                one_hot_encoded_size=one_hot_encoded_size,
                                                                                sub_sequence_length=sub_sequence_length)

    # make and train model
    start_time2 = time.time()

    rbns_classifier = RBNS_Classifier(use_load_model=False, one_hot_encoded_size=one_hot_encoded_size,
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

    # dlincb_utils.write_predictions_to_file(rna_predictions, protein_number=8)
    # dlincb_utils.calculate_and_graph_pearson_for_all()
    # Calculate and print total runtime
    print("Total runtime of a single protein:", (time.time() - start_time), "seconds")

    print("END!")
