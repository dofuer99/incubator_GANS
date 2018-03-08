from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from keras.models import load_model

import sys
import csv
import numpy as np

import data_transform 
import pandas as pd
#from sklearn.metrics import log_loss

class GAN():
    def __init__(self):
        self.train_cat = True 
        self.pair_size = 30
        self.img_rows = 8 * self.pair_size
        self.img_cols = 1
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.best_epoch = 0

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (100,)
        
        model = Sequential()

        model.add(Dense(128, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        #np.prod calulate the multiplications of the elements
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        #model.add(Dense(np.prod(self.img_shape), activation='sigmoid'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        
        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):
        results_history = []
        best_acc = 1
        # Load the dataset
        df = pd.read_csv('bank-full.csv', header = 0, sep = ';')

        try1 = data_transform.encode(df)
        raw = try1.encode_begin()
        
        #wrtie out the initial transformed data for comparison
        with open('gen_data/initial_transformed_data.csv', 'w', newline='') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            row = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan']
            wr.writerow(row)
            for row in raw:
                wr.writerow(row) 

        try2 = data_transform.scaled(raw)
        scaled, self.cache = try2.scale_begin()
        
        if self.train_cat:
            self.new_length = len(scaled) // self.pair_size
            dropped_scaled = scaled[ : self.new_length * self.pair_size]
            new_scaled = dropped_scaled.reshape(self.new_length, -1)
            input_data = new_scaled
        else:
            input_data = scaled       
            
#        with open('test.csv') as csvDataFile:
#            csvReader = csv.reader(csvDataFile)
#            test_new = []
#            for row in csvReader:
#                row_new = []
#                for element in row:
#                    row_new.append(float(element))
#                test_new.append(row_new)
#                
#        input_data = np.array(test_new)        
        #split the data into train set and test set (0.8, 0.2)        
        data_len = len(input_data) 
        data_train_len = int(data_len * 0.8)
        idx_train = np.random.randint(0, data_len, data_train_len)
        idx_test = [i for i in range(data_len) if i not in idx_train]
        data_train = input_data[idx_train]
        data_test = input_data[idx_test]
        
        #expand the data dimension to (N,D,1,1)
        half_batch = int(batch_size / 2)
        data_train = np.expand_dims(data_train, axis=2)
        X_train = np.expand_dims(data_train, axis=3)
        
        data_test = np.expand_dims(data_test, axis=2)
        X_test = np.expand_dims(data_test, axis=3)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)
            
            # ---------------------
            #  Test Accuracy Calculation
            # ---------------------
            # real data prediction
            real_len = X_test.shape[0]
            pred_real = self.discriminator.predict(X_test)
            #push the intermediate value to the extreme
            real_acc = np.sum([pred_real > 0.5]) * 1.0 / real_len
            
            # fake data prediction
            noise = np.random.normal(0, 1, (real_len, 100))
            X_fake_test = self.generator.predict(noise)
            #round the fake data and then scale it back
#            pred_fake = np.squeeze(X_fake_test, axis=2)
#            try3 = data_transform.unscaled(pred_fake, self.cache) 
#            unscaled = try3.unscale_begin()
#            try4 = data_transform.round_data(unscaled)
#            rounded_data = try4.round_all()
#            
#            try2 = data_transform.scaled(rounded_data)
#            final_fake_data, _ = try2.scale_begin()
#            final_fake_data = np.expand_dims(final_fake_data, axis=3)
            #print (final_fake_data.shape)
            pred_fake = self.discriminator.predict(X_fake_test)
            #push the intermediate value to the extreme
            fake_acc = np.sum([pred_fake < 0.5]) * 1.0 / real_len       
            
            pred_acc = 0.5 * (real_acc + fake_acc) 
            
            #save the best model
            if pred_acc < best_acc:
                best_acc = pred_acc
                self.generator.save('gen_data/best_model.h5')
                self.best_epoch = epoch
                
            if epoch == epochs - 1:
                self.generator.save('gen_data/last_model.h5')

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [Test Acc: %.2f%%]" % (epoch, d_loss[0], 100*d_loss[1], g_loss, 100*pred_acc))
                       
            if epoch % save_interval == 0:
                self.save_data(epoch)
                
            if epoch % 1 == 0 and epoch > 1:
                results_history.append([epoch, d_loss[0], 100*d_loss[1], g_loss, pred_acc])
                
            if epoch == epochs - 1:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                
                xxx = [row[0] for row in results_history]
                D_accuracy = [row[2] for row in results_history]
                D_loss = [row[1] for row in results_history]
                G_loss = [row[3] for row in results_history]
                P_acc = [row[4] for row in results_history] 
                
                lns1 = ax.plot(xxx, P_acc, 'g', label='test accuracy')
                ax2 = ax.twinx()
                lns2 = ax2.plot(xxx, D_loss, 'b', label='discrminative loss')
                lns3 = ax2.plot(xxx, G_loss, 'r', label='generative loss')

                lns = lns1 + lns2 +lns3
                labs = [l.get_label() for l in lns]
                ax.legend(lns,labs,loc=0)
                
                ax.set_ylabel('Acurracy (100%)')
                ax2.set_ylabel('Loss')

                ax2.set_ylim(0,4)
                ax.set_ylim(0,1.5)
                fig.savefig('plot_history')
                
                fig1 = plt.figure()
                ax3 = fig1.add_subplot(111)    
                ax3.plot(xxx, D_accuracy, 'g', label='discriminative accuracy')
                ax3.set_ylabel('Test Acurracy (100%)')
                fig1.savefig('plot_accuracy')
        print ('The best model is obtained in epoch %d.' % self.best_epoch)
                
                       
    def save_data(self, epoch):
        gen_size = 1000  #the number of fake data you want to generate
        noise = np.random.normal(0, 1, (gen_size, 100))
        #self.generator = load_model('saved_model/best_model.h5')  the training process will fail if we do not comment this
        gen_data = self.generator.predict(noise)
        gen_data_new = np.squeeze(gen_data, axis=2)
        if self.train_cat:
            gen_data_temp = gen_data_new.reshape(gen_size * self.pair_size, -1)
        else:
            gen_data_temp = gen_data_new
        #unscale the data
        try3 = data_transform.unscaled(gen_data_temp, self.cache) 
        unscaled = try3.unscale_begin()
        #round the data
        try4 = data_transform.round_data(unscaled)
        rounded_data = try4.round_all()
        with open('gen_data/gen_data_%d.csv' % epoch, 'w', newline='') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            row = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan']
            wr.writerow(row)
            for row in rounded_data:
                wr.writerow(row)       
        

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=2001, batch_size=128, save_interval=100)






