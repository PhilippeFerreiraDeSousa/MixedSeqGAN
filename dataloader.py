import numpy as np
import pandas as pd


class Gen_Data_loader():
    def __init__(self, batch_size, seq_length):
        self.batch_size = batch_size
        self.token_stream = []
        self.seq_length = seq_length

    def create_batches(self, data_file):
        self.token_stream = []
        df = pd.read_hdf(data_file)
        count = len(df)

        for i in range(count // self.seq_length):
            self.token_stream.append(df.values[i * self.seq_length:(i+1) * self.seq_length])

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader():
    def __init__(self, batch_size, seq_length):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.seq_length = seq_length

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []

        df_pos = pd.read_hdf(positive_file)
        count_pos = len(df_pos)
        for i in range(int(count_pos / self.seq_length)):
            positive_examples.append(df_pos.values[i * self.seq_length:(i + 1) * self.seq_length])

        df_neg = pd.read_hdf(negative_file)
        count_neg = len(df_neg)
        for i in range(int(count_neg / self.seq_length)):
            negative_examples.append(df_neg.values[i * self.seq_length:(i + 1) * self.seq_length])

        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

