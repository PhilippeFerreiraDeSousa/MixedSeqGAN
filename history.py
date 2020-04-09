import pickle
import numpy as np

class History:
    def __init__(self,
                 HISTORY_FILE,
                 DIS_PRETRAINING_FAKE_SAMPLE_COUNT,
                 DIS_PRETRAINING_UPDATE_COUNT,
                 BATCH_SIZE,
                 SEQ_LENGTH,
                 generated_num,
                 EPOCH_COUNT,
                 GEN_TRAINING_UPDATE_COUNT,
                 DIS_TRAINING_FAKE_SAMPLE_COUNT,
                 DIS_TRAINING_UPDATE_COUNT):
        self.HISTORY_FILE = HISTORY_FILE
        self.DIS_PRETRAINING_FAKE_SAMPLE_COUNT = DIS_PRETRAINING_FAKE_SAMPLE_COUNT
        self.DIS_PRETRAINING_UPDATE_COUNT = DIS_PRETRAINING_UPDATE_COUNT
        self.BATCH_SIZE = BATCH_SIZE
        self.SEQ_LENGTH = SEQ_LENGTH
        self.generated_num = generated_num
        self.EPOCH_COUNT = EPOCH_COUNT
        self.GEN_TRAINING_UPDATE_COUNT = GEN_TRAINING_UPDATE_COUNT
        self.DIS_TRAINING_FAKE_SAMPLE_COUNT = DIS_TRAINING_FAKE_SAMPLE_COUNT
        self.DIS_TRAINING_UPDATE_COUNT = DIS_TRAINING_UPDATE_COUNT

        self.pre_training_loss = []
        self.generator_categorical_loss = []
        self.generator_continuous_loss = []
        self.discriminator_loss = []


    @property
    def generator_loss(self):
        return np.array(self.generator_categorical_loss) + np.array(self.generator_continuous_loss)

    def save(self):
        with open(self.HISTORY_FILE, 'wb') as history_file:
            pickle.dump(self, history_file)

    @staticmethod
    def load(HISTORY_FILE):
        with open(HISTORY_FILE, "rb") as history_file:
            return pickle.load(history_file)
