import os
import time
import struct
import numpy as np
from Layer import FullyConnectedLayer, ReluLayer, SoftmaxLossLayer

DATA_DIR = './MNIST'
MODEL_FILE = 'weight.npy'
HIDDEN_1_DEEP = 64
HIDDEN_2_DEEP = 32

class HWNR_Model(object):
    def __init__(self, batch_size=100, learning_rate=0.01, epoch_num=10, input_size=784, output_class=10):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.input_size = input_size
        self.output_class = output_class
        self.hiden_1_size = HIDDEN_1_DEEP
        self.hiden_2_size = HIDDEN_2_DEEP

    def load_data(self, minist_dir_path):
        print('----------loading MINIST data----------')
        train_data = self._load_minist(os.path.join(
            minist_dir_path, "train-images-idx3-ubyte"), 'data')
        train_label = self._load_minist(os.path.join(
            minist_dir_path, "train-labels-idx1-ubyte"), 'label')
        test_data = self._load_minist(os.path.join(
            minist_dir_path, "t10k-images-idx3-ubyte"), 'data')
        test_label = self._load_minist(os.path.join(
            minist_dir_path, "t10k-labels-idx1-ubyte"), 'label')
        # self.train_data = self._normalization(train_data)
        # self.test_data = self._normalization(test_data)
        self.train_data = train_data
        self.test_data = test_data
        self.train_label = np.eye(10)[train_label.flatten()]
        self.test_label = np.eye(10)[test_label.flatten()]

        print('\ttrain_data shape : ' + str(self.train_data.shape) +
              '\ttrain_label shape : ' + str(self.train_label.shape))
        print('\ttest_data  shape : ' + str(self.test_data.shape) +
              '\ttest_label  shape : ' + str(self.test_label.shape))

    def shuffle(self):
        shuffle_ix = np.random.permutation(np.arange(self.train_data.shape[0]))
        self.train_data = self.train_data[shuffle_ix]
        self.train_label = self.train_label[shuffle_ix]

    def build_model(self,):
        
        print('----------build model------------')
        self.fc1 = FullyConnectedLayer(self.input_size, self.hiden_1_size)
        self.relu1 = ReluLayer()

        self.fc2 = FullyConnectedLayer(self.hiden_1_size, 
                                       self.hiden_2_size)
        self.relu2 = ReluLayer()

        self.fc3 = FullyConnectedLayer(self.hiden_2_size, self.output_class)
        self.softmax = SoftmaxLossLayer()

        self.model_layer_list = [self.fc1, self.relu1,
                                 self.fc2, self.relu2,
                                 self.fc3, self.softmax]
        self.update_layer_list = [self.fc1, self.fc2, self.fc3]

    def load_model(self, param_file):
        print('----------loading parameters from file ' + param_file)
        params = np.load(param_file, allow_pickle=True).item()
        self.fc1.load_param(params['fc1_weight'], params['fc1_bias'])
        self.fc2.load_param(params['fc2_weight'], params['fc2_bias'])
        self.fc3.load_param(params['fc3_weight'], params['fc3_bias'])

    def save_model(self, param_file):
        print('----------saving parameters to file' + param_file)
        params = {}
        params['fc1_weight'], params['fc1_bias'] = self.fc1.save_param()
        params['fc2_weight'], params['fc2_bias'] = self.fc2.save_param()
        params['fc3_weight'], params['fc3_bias'] = self.fc3.save_param()
        np.save(param_file, params)

    def model_forward(self, input_X):
        fc1_output = self.fc1.forward(input_X)
        relu1_output = self.relu1.forward(fc1_output)
        fc2_output = self.fc2.forward(relu1_output)
        relu2_output = self.relu2.forward(fc2_output)
        fc3_output = self.fc3.forward(relu2_output)
        softmax_output = self.softmax.forward(fc3_output)
        return softmax_output

    def model_backward(self):
        softmax_grad = self.softmax.backward()
        fc3_grad = self.fc3.backward(softmax_grad)
        relu2_grad = self.relu2.backward(fc3_grad)
        fc2_grad = self.fc2.backward(relu2_grad)
        relu1_grad = self.relu1.backward(fc2_grad)
        self.fc1.backward(relu1_grad)

    def model_update(self, learning_rate):
        for layer in self.update_layer_list:
            layer.update_param(learning_rate)

    def train(self,):
        batch_num = self.train_data.shape[0] // self.batch_size
        print('----------start training----------')
        for epoch_idx in range(self.epoch_num):
            self.shuffle()
            for batch_idx in range(batch_num):
                batch_data = self.train_data[batch_idx *
                                             self.batch_size:(batch_idx + 1) * self.batch_size, :]
                batch_label = self.train_label[batch_idx *
                                               self.batch_size:(batch_idx + 1) * self.batch_size, :]

                self.model_forward(batch_data)
                batch_loss = self.softmax.loss(batch_label)
                self.model_backward()
                self.model_update(self.learning_rate)
                if batch_idx % 100 == 0:
                    print('epoch %d, batch %d, loss: %.6f' %
                          (epoch_idx, batch_idx / 100, batch_loss))

    def evaluation(self):
        prediction = np.zeros([self.test_data.shape[0], 10])
        for idx in range(self.test_data.shape[0] // self.batch_size):
            batch_data = self.test_data[idx *
                                        self.batch_size:(idx + 1) * self.batch_size, :]
            batch_prediction = self.model_forward(batch_data)
            prediction[idx*self.batch_size:(idx+1)
                       * self.batch_size, :] = batch_prediction
        accuracy = np.mean(np.argmax(prediction, axis=1) ==
                           np.argmax(self.test_label, axis=1))
        print('Accuracy in test set : %f' % accuracy)

    def _normalization(self, data):
        
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def _load_minist(self, file_path, type='data'):

        binary_file = open(file_path, 'rb')
        binary_data = binary_file.read()
        binary_file.close()

        if type == 'data':
            fmt_header = '>iiii'
            _, num_images, num_rows, num_cols = struct.unpack_from(
                fmt_header, binary_data, 0)
        else:
            fmt_header = '>ii'
            num_rows, num_cols = 1, 1
            _, num_images = struct.unpack_from(fmt_header, binary_data, 0)

        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from(
            '>' + str(data_size) + 'B', binary_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
        return mat_data


if __name__ == '__main__':
    model = HWNR_Model()
    model.load_data(DATA_DIR)
    model.build_model()
    model.train()
    model.save_model(MODEL_FILE)
    model.load_model(MODEL_FILE)
    model.evaluation()
