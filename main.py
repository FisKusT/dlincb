# Project in Deep Learning in Computational Biology course.

import dlincb_utils
from nn_model import RBNS_Classifier
import time


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Measure start time
    start_time = time.time()

    # Parse RBNS data for protein
    seq_file_list = [
        "C:/Users/sorte/Desktop/BIU/DL in CB/project/RBNS_training/RBP1_input.seq",
        "C:/Users/sorte/Desktop/BIU/DL in CB/project/RBNS_training/RBP1_1300nM.seq"
    ]

    # Parse RNA data
    rna_file = "C:/Users/sorte/Desktop/BIU/DL in CB/project/RNAcompete_sequences.txt"
    rna_sequences = dlincb_utils.extract_data_from_file_to_array(rna_file)

    x_one_hot, y_labels = dlincb_utils.create_train_and_test_set_from_seq_files(seq_file_list)

    # make and train model
    rbns_classifier = RBNS_Classifier(use_load_model=True)
    rbns_classifier.train_model(x_one_hot, y_labels)

    # Parse RNA data
    rna_file = "C:/Users/sorte/Desktop/BIU/DL in CB/project/RNAcompete_sequences.txt"
    rna_sequences = dlincb_utils.extract_data_from_file_to_array(rna_file)

    rna_sequences_one_hot_vector = dlincb_utils.convert_sequences_to_one_hot_vector(rna_sequences)

    # get the real value predictions
    # real_rna_values_file = "C:/Users/sorte/Desktop/BIU/DL in CB/project/RNCMPT_training/RBP1.txt"
    # real_rna_values = np.array(dlincb_utils.extract_data_from_file_to_array(real_rna_values_file),
    #                            dtype=np.float32)

    # calculate pearson correlation for
    # correlation1 = scipy.stats.pearsonr(real_rna_values, rna_predictions)
    # correlation2 = scipy.stats.pearsonr(real_rna_values, rna_predictions2)
    # print("Pearson correlation1", correlation1)
    # print("Pearson correlation2", correlation2)
    # Calculate and print total runtime
    print("Total runtime of a single protein:", (time.time() - start_time), "seconds")

    print("END!")
