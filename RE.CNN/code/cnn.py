##########################################################################################################
###                                         CNN.py                                              ###
##########################################################################################################
"""
This is a CNN for relation classification within a sentence. The architecture is based on:

Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou and Jun Zhao, 2014, Relation Classification via Convolutional Deep Neural Network

Performance (without hyperparameter optimization):
Accuracy: 0.7943
Macro-Averaged F1 (without Other relation):  0.7612

Performance Zeng et al.
Macro-Averaged F1 (without Other relation): 0.789

Code was tested with:
- Python 2.7 & Python 3.6
- Theano 0.9.0 & TensorFlow 1.2.1
- Keras 2.0.5
"""

from __future__ import print_function
import numpy as np
import gzip
import sys
import pickle as pkl
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.regularizers import Regularizer
from keras.preprocessing import sequence
np.random.seed(1337)  # for reproducibility

##################  function  #################
def getPrecision(pred_test, y_test, targetLabel):
    # Precision for non-vague
    targetLabelCount = 0
    correctTargetLabelCount = 0
    
    for idx in range(len(pred_test)):
        if pred_test[idx] == targetLabel:
            targetLabelCount += 1
            
            if pred_test[idx] == y_test[idx]:
                correctTargetLabelCount += 1
    
    if correctTargetLabelCount == 0:
        return 0
    
    return float(correctTargetLabelCount) / targetLabelCount


def predict_classes(prediction):
    return prediction.argmax(axis=-1)



##################  parameters  #################
batch_size = 64
nb_filter = 100
filter_length = 3
hidden_dims = 100
nb_epoch = 100
pos_dims = 50

##################  read data  #################
print("Loading dataset ...")
f = gzip.open('data/sem-relations.pkl.gz', 'rb')
data = pkl.load(f)
f.close()

embeddings = data['wordEmbeddings']
y_train, sent_train, pos1_train, pos2_train = data['train_set']
y_test, sent_test, pos1_test, pos2_test  = data['test_set']
max_pos = max(np.max(pos1_train), np.max(pos2_train)) + 1
n_out = max(y_train) + 1
# train_y_cat = np_utils.to_categorical(y_train, n_out)
max_sent_len = sent_train.shape[1]
print("Dimension sent_train: ", sent_train.shape)
print("Dimension pos1_train: ", pos1_train.shape)
print("Dimension y_train: ", y_train.shape)
print("Dimension sent_test: ", sent_test.shape)
print("Dimension pos1_test: ", pos1_test.shape)
print("Dimension y_test: ", y_test.shape)
print("Dimension Embeddings: ", embeddings.shape)


##################  CNN model  #################
# embedding layers
words_input = Input(shape=(max_sent_len,), dtype='int32', name='words_input')
words = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)(words_input)

dist1_input = Input(shape=(max_sent_len,), dtype='int32', name='dist1_input')
dist1 = Embedding(max_pos, pos_dims)(dist1_input)

dist2_input = Input(shape=(max_sent_len,), dtype='int32', name='dist2_input')
dist2 = Embedding(max_pos, pos_dims)(dist2_input)

output = concatenate([words, dist1, dist2])

# convolution layer
output = Convolution1D(filters=nb_filter, kernel_size=filter_length, padding='same', activation='tanh', strides=1)(output)

# we use standard max over time pooling
output = GlobalMaxPooling1D()(output)
output = Dropout(0.25)(output)
output = Dense(n_out, activation='softmax')(output)

model = Model(inputs=[words_input, dist1_input, dist2_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()


##################  training  #################
print("Start training ...")
max_prec, max_rec, max_acc, max_f1 = 0, 0, 0, 0

for epoch in range(nb_epoch):       
    model.fit([sent_train, pos1_train, pos2_train], y_train, batch_size=batch_size, verbose=2, epochs=1)   
    pred_test = predict_classes(model.predict([sent_test, pos1_test, pos2_test], verbose=0))
    
    dctLabels = np.sum(pred_test)
    totalDCTLabels = np.sum(y_test)
   
    acc =  np.sum(pred_test == y_test) / float(len(y_test))
    max_acc = max(max_acc, acc)
    print("Accuracy: %.4f (max: %.4f)" % (acc, max_acc))

    f1Sum = 0
    f1Count = 0
    for targetLabel in range(1, max(y_test)):        
        prec = getPrecision(pred_test, y_test, targetLabel)
        recall = getPrecision(y_test, pred_test, targetLabel)
        f1 = 0 if (prec+recall) == 0 else 2*prec*recall/(prec+recall)
        f1Sum += f1
        f1Count +=1    
            
    macroF1 = f1Sum / float(f1Count)    
    max_f1 = max(max_f1, macroF1)
    print("Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1))