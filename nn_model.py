# The neural network model
# each protein has a different model

import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.src.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.models import load_model
import time


class RBNS_Classifier:
    def __init__(self, use_load_model=False):
        self.train_history = None
        self.model_path = "C:/Users/sorte/Desktop/BIU/DL in CB/project/bestmodel.h5"
        if use_load_model:
            self.model = load_model(self.model_path)
        else:
            self.model = self.build_model()

    # TODO:
    # def predict_rna_sequences(self, rna_sequences):
    #     # Measure start time
    #     start_time = time.time()
    #     # generate from RNA sequence- subsequences of len 20 one hot vectors
    #     rna_sub_sequences_list = []
    #     batches_indexes_list = [0]
    #     for rna_sequence in rna_sequences:
    #         rna_sub_sequences = convert_rna_sequence_to_batches(rna_sequence)
    #         batches_index = batches_indexes_list[-1] + len(rna_sub_sequences)
    #         rna_sub_sequences_list = np.concatenate([rna_sub_sequences_list, rna_sub_sequences])
    #         batches_indexes_list.append(batches_index)
    #     rna_sub_sequences_one_hot = convert_sequences_to_one_hot_vector(rna_sub_sequences_list)
    #     predictions = self.model.predict(rna_sub_sequences_one_hot)
    #     # get max predictions of every RNA
    #     max_predictions_list = []
    #     for i in range(len(batches_indexes_list) - 1):
    #         max_prediction = np.amax(predictions[batches_indexes_list[i]:batches_indexes_list[i + 1]])
    #         max_predictions_list.append(max_prediction * 3 - 1)
    #     # Measure end time
    #     print("Total runtime of predict_rna_sequence:", (time.time() - start_time), "seconds")
    #     return np.array(max_predictions_list, dtype=np.float32)

    def train_model(self, x, y):
        print("Start train model")
        # Measure start time
        start_time = time.time()
        # make training and validation set
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.2,
                                                            shuffle=True,
                                                            random_state=123)
        # train the model
        self.train_history = self.model.fit(x_train, y_train,
                                            batch_size=64,
                                            epochs=30,
                                            validation_data=(x_test, y_test),
                                            shuffle=True,
                                            verbose=1,
                                            callbacks=[
                                                ModelCheckpoint(
                                                    self.model_path,
                                                    monitor='val_loss',
                                                    save_best_only=True),
                                                EarlyStopping(monitor='val_loss',
                                                              patience=3,
                                                              restore_best_weights=True)
                                            ])
        # Measure end time
        print("Total runtime of train_model:", (time.time() - start_time), "seconds")
        return self.train_history

    def build_model(self):
        # create keras sequential model
        model = Sequential()
        # 1D convolutional layer
        model.add(Conv1D(filters=512,
                         kernel_size=8,
                         strides=1,
                         kernel_initializer='RandomNormal',
                         activation='relu',
                         kernel_regularizer=l2(5e-3),
                         input_shape=(20, 4),
                         use_bias=True,
                         bias_initializer='RandomNormal'))

        # Max pooling
        model.add(MaxPooling1D(pool_size=5))

        # Flatten the output before fully connected layers
        model.add(Flatten())

        # Fully Connected Layers
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))

        # Output layer with sigmoid activation for binary classification
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, amsgrad=False),
                      loss=binary_crossentropy)
        return model
