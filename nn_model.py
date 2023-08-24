# Tal Fiskus (ID: 208423707)
# Yoel Benabou (ID: 342875648)

# The neural network model
# each protein has a different model

import dlincb_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout, LeakyReLU
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.src.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.model_selection import train_test_split
from keras.models import load_model
import time


class RBNS_Classifier:
    def __init__(self, use_load_model=False, one_hot_encoded_size = 4, sub_sequence_length=20):
        """
        Initialize the model class.
        :param use_load_model: use the load model or build it
        """
        self.train_history = None
        self.model_path = "bestmodel.h5"
        if use_load_model:
            self.model = load_model(self.model_path)
        else:
            self.model = self.build_model(one_hot_encoded_size=one_hot_encoded_size, sub_sequence_length=sub_sequence_length)
        self.model.summary()

    def predict_for_batch(self, data, max_combinaisons, one_hot_encoded_size, sub_sequence_length):
        """
        Do the prediction for all the batch.
        :param sub_sequence_length:
        :param one_hot_encoded_size:
        :param data:
        :param max_combinaisons:
        :return:
        """
        batch_size = len(data)
        reshaped_input = data.reshape(-1, sub_sequence_length, one_hot_encoded_size)

        # Make predictions for all 19 arrays in one pass
        predictions = self.model.predict(reshaped_input)

        # Reshape the predictions to match the original shape of (batch_size, max_combinations)
        predictions = predictions.reshape(batch_size, max_combinaisons)

        max_preds = dlincb_utils.clean_padding_from_prediction(data, predictions, batch_size, max_combinaisons)
        return np.max(max_preds, axis=1)

    def do_prediction(self, all_subsequences, max_combinaisons, one_hot_encoded_size, sub_sequence_length):
        """
        Function to do the prediction for all sequences
        :param all_subsequences:
        :param max_combinaisons:
        :param one_hot_encoded_size:
        :param sub_sequence_length:
        :return:
        """
        predictions = []
        batch_size = 512
        num_batches = len(all_subsequences) // batch_size
        remaining_sequences = len(all_subsequences) % batch_size
        for batch_index in range(num_batches):
            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            batch = all_subsequences[start_index:end_index]

            self.process_prediction_by_batch(batch, max_combinaisons, predictions, batch_index, num_batches,
                                             one_hot_encoded_size=one_hot_encoded_size, sub_sequence_length=sub_sequence_length)

        if remaining_sequences > 0:
            start_index = num_batches * batch_size
            remaining_batch = all_subsequences[start_index:]
            self.process_prediction_by_batch(remaining_batch, max_combinaisons, predictions, num_batches, num_batches,
                                             one_hot_encoded_size=one_hot_encoded_size, sub_sequence_length=sub_sequence_length)

        return predictions

    def process_prediction_by_batch(self, batch, max_combinaisons, predictions, batch_index, num_batches, one_hot_encoded_size, sub_sequence_length):
        """
        Do the prediction for batch
        :param batch:
        :param max_combinaisons:
        :param predictions:
        :param batch_index:
        :param num_batches:
        :param one_hot_encoded_size:
        :param sub_sequence_length:
        :return:
        """
        print("Prediction : " + str((batch_index + 1)) + " / " + str(num_batches))
        encoded_batches = dlincb_utils.one_hot_encoding_rna_sequences_by_batch(batch, max_combinaisons,
                                                                               one_hot_encoded_size=one_hot_encoded_size,
                                                                               sub_sequence_length=sub_sequence_length)
        predictions.extend(self.predict_for_batch(encoded_batches, max_combinaisons=max_combinaisons,
                                                  one_hot_encoded_size=one_hot_encoded_size, sub_sequence_length=sub_sequence_length))

    def train_model(self, x, y):
        """
        Train the model.
        :param x: x data
        :param y: labels data
        :return: train history
        """
        print("Start train model")
        # Measure start time
        start_time = time.time()
        # make training and validation set
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.2,
                                                            shuffle=True,
                                                            random_state=42)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
        ]

        # train the model
        self.train_history = self.model.fit(x_train, y_train,
                                            batch_size=4096,
                                            epochs=30,
                                            validation_data=(x_test, y_test),
                                            shuffle=True,
                                            verbose=1,
                                            callbacks=callbacks)
        # Measure end time
        print("Total runtime of train_model:", (time.time() - start_time), "seconds")
        return self.train_history

    def build_model(self, one_hot_encoded_size, sub_sequence_length):
        """
            Build our sequential model.
        """
        # create keras sequential model
        model = Sequential()
        # 1D convolutional layer
        model.add(Conv1D(filters=512,
                         kernel_size=8,
                         strides=1,
                         kernel_initializer='glorot_normal', # xavier initialization
                         activation=LeakyReLU(alpha=0.01),
                         kernel_regularizer=l2(5e-3),
                         input_shape=(sub_sequence_length, one_hot_encoded_size),
                         use_bias=True,
                         bias_initializer='RandomNormal'))

        # Max pooling
        model.add(MaxPooling1D(pool_size=5))

        # Flatten the output before fully connected layers
        model.add(Flatten())

        # Fully Connected Layers
        model.add(Dense(64, activation=LeakyReLU(alpha=0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(64, activation=LeakyReLU(alpha=0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(64, activation=LeakyReLU(alpha=0.01)))
        model.add(Dense(32, activation=LeakyReLU(alpha=0.01)))
        model.add(Dense(16, activation=LeakyReLU(alpha=0.01)))

        # Output layer with sigmoid activation for binary classification
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, amsgrad=False),
                      loss=binary_crossentropy)
        return model
