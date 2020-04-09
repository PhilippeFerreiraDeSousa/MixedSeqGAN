import numpy as np

import tensorflow as tf
import pandas as pd
import os
from history import History
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from config import DIS_PRETRAINING_FAKE_SAMPLE_COUNT,\
    DIS_PRETRAINING_UPDATE_COUNT, \
    EPOCH_COUNT, \
    GEN_TRAINING_UPDATE_COUNT, \
    DIS_TRAINING_FAKE_SAMPLE_COUNT, \
    DIS_TRAINING_UPDATE_COUNT


#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 8 # embedding dimension
HIDDEN_DIM = 8 # hidden state dimension of lstm cell
SEQ_LENGTH = 50 # sequence length
START_TOKEN = [0., 0., 0]
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 16
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [50, 100, 100, 100, 100, 50, 50, 50, 50, 50, 80, 80]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
positive_file = 'save/CDS_Preprocessed_Data_merged.h5.gz'
negative_file = 'save/generator_sample.h5.gz'
generated_num = 37212 // SEQ_LENGTH # generate the same number of negative, in reality we get only 1856 sequences instead of 1860 ???


# Loop iteration counts
# DIS_PRETRAINING_FAKE_SAMPLE_COUNT = 50
# DIS_PRETRAINING_UPDATE_COUNT = 3
#
# EPOCH_COUNT = 200
# GEN_TRAINING_UPDATE_COUNT = 1   # Let this at 1 typically
# DIS_TRAINING_FAKE_SAMPLE_COUNT = 5
# DIS_TRAINING_UPDATE_COUNT = 3

# Files
MODEL_FILE = "save/model.ckpt"
RANDOM_MODEL_FILE = "save/initial_random_model.ckpt"
HISTORY_FILE = "save/history.pkl"


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    data = []
    for sequence in generated_samples:
        data.extend(sequence)
    df = pd.DataFrame(data=data)
    df.to_hdf(output_file, key="CDS", complib="zlib", complevel=9, mode='w')

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    history = History(HISTORY_FILE,
                      DIS_PRETRAINING_FAKE_SAMPLE_COUNT,
                      DIS_PRETRAINING_UPDATE_COUNT,
                      BATCH_SIZE,
                      SEQ_LENGTH,
                      generated_num,
                      EPOCH_COUNT,
                      GEN_TRAINING_UPDATE_COUNT,
                      DIS_TRAINING_FAKE_SAMPLE_COUNT,
                      DIS_TRAINING_UPDATE_COUNT)

    gen_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH)
    vocab_size = 112 # np.array([2, 3, 2, 4, 4]) + np.array([1, 1, 1, 1, 1])
    dis_data_loader = Dis_dataloader(BATCH_SIZE, SEQ_LENGTH)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    if os.path.exists(MODEL_FILE + ".index"):
        saver.restore(sess, MODEL_FILE)
        print("Model restored.")
    else:
        save_random_path = saver.save(sess, RANDOM_MODEL_FILE)
        print("Initial random model saved in path: %s" % save_random_path)

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    gen_data_loader.create_batches(positive_file)

    print('Start pre-training discriminator...')
    # Train 3 epoch on the generated data and do this for 50 times
    for i in range(DIS_PRETRAINING_FAKE_SAMPLE_COUNT):
        print(" > Fake sample " + str(i) + "/" + str(DIS_PRETRAINING_FAKE_SAMPLE_COUNT))
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for j in range(DIS_PRETRAINING_UPDATE_COUNT):
            dis_data_loader.reset_pointer()
            print("   > Pass " + str(j) + "/" + str(DIS_PRETRAINING_UPDATE_COUNT))
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _, loss = sess.run([discriminator.train_op, discriminator.loss], feed)
                history.pre_training_loss.append(loss)
                # print("     > Discriminator params updated")

    rollout = ROLLOUT(generator, 0.8)
    history.save()

    print('#########################################################################')
    print('Start Adversarial Training...')
    for epoch_num in range(EPOCH_COUNT):
        # Train the generator for one step
        print(" > Epoch " + str(epoch_num) + "/" + str(EPOCH_COUNT))
        for it in range(GEN_TRAINING_UPDATE_COUNT):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _, categorical_loss, continuous_loss = sess.run([generator.g_updates, generator.categorical_loss, generator.continuous_loss], feed_dict=feed)
            history.generator_categorical_loss.append(categorical_loss)
            history.generator_continuous_loss.append(continuous_loss)
            print("   > Generator params updated")

        # Update roll-out parameters
        rollout.update_params()
        print("   > Rollout params updated")

        # Train the discriminator
        for i in range(DIS_TRAINING_FAKE_SAMPLE_COUNT):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
            print("   > Discriminator training sample " + str(i) + "/" + str(DIS_TRAINING_FAKE_SAMPLE_COUNT))

            for j in range(DIS_TRAINING_UPDATE_COUNT):
                dis_data_loader.reset_pointer()
                print("     > Pass " + str(j) + "/" + str(DIS_TRAINING_UPDATE_COUNT))
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _, loss = sess.run([discriminator.train_op, discriminator.loss], feed)
                    history.discriminator_loss.append(loss)
                    # print("     > Discriminator params updated")

        if epoch_num % 10 == 0:
            history.save()
            save_path = saver.save(sess, MODEL_FILE)
            print("Intermediate model at epoch saved in path: %s" % save_path)

    history.save()
    save_path = saver.save(sess, MODEL_FILE)
    print("Final model saved in path: %s" % save_path)


if __name__ == '__main__':
    main()
