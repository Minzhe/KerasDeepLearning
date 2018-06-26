##########################################################################################################
###                                         preprocess.py                                              ###
##########################################################################################################
'''
The file preprocesses the files/train.txt and files/test.txt files.

It requires the dependency based embeddings by Levy et al.. 
Download them from his website and change the embeddingsPath variable 
in the script to point to the unzipped deps.words file.
'''

from __future__ import print_function
import numpy as np
import gzip
import os
import sys
import pickle as pkl

####################################   function   #####################################
def createMatrices(file, word2Idx, max_sent_len=100):
    '''
    Creates matrices for the events and sentence for the given file
    '''
    # Mapping of the labels to integers
    labelsMapping = {'Other':0,
                     'Message-Topic(e1,e2)':1, 'Message-Topic(e2,e1)':2, 
                     'Product-Producer(e1,e2)':3, 'Product-Producer(e2,e1)':4, 
                     'Instrument-Agency(e1,e2)':5, 'Instrument-Agency(e2,e1)':6, 
                     'Entity-Destination(e1,e2)':7, 'Entity-Destination(e2,e1)':8,
                     'Cause-Effect(e1,e2)':9, 'Cause-Effect(e2,e1)':10,
                     'Component-Whole(e1,e2)':11, 'Component-Whole(e2,e1)':12, 
                     'Entity-Origin(e1,e2)':13, 'Entity-Origin(e2,e1)':14,
                     'Member-Collection(e1,e2)':15, 'Member-Collection(e2,e1)':16,
                     'Content-Container(e1,e2)':17, 'Content-Container(e2,e1)':18}

    dist_map = {'PADDING': 0, 'min': 1, 'max': 2}
    min_dist, max_dist = -30, 30
    for dis in range(min_dist, max_dist+1):
        dist_map[dis] = len(dist_map)
    
    labels, pos1_mat, pos2_mat, tokenId_mat = [], [], [], []
    
    for line in open(file):
        splits = line.strip().split('\t')
        label_, pos1_, pos2_, tokens_ = splits[0], splits[1], splits[2], splits[3].split(' ')
        tokenIds_, pos1_val_, pos2_val_ = np.zeros(max_sent_len), np.zeros(max_sent_len), np.zeros(max_sent_len)
        
        # loop through each token in the sentense
        for i_ in range(0, min(max_sent_len, len(tokens_))):
            tokenIds_[i_] = getWordIdx(tokens_[i_], word2Idx)
            dist1, dist2 = i_ - int(pos1_), i_ - int(pos2_)
            pos1_val_[i_] = mapDist(dist=dist1, dist_map=dist_map, min_dist=min_dist, max_dist=max_dist)
            pos2_val_[i_] = mapDist(dist=dist2, dist_map=dist_map, min_dist=min_dist, max_dist=max_dist)
            
        tokenId_mat.append(tokenIds_)
        pos1_mat.append(pos1_val_)
        pos2_mat.append(pos2_val_)
        labels.append(labelsMapping[label_])
        
    return np.array(labels, dtype='int32'), np.array(tokenId_mat, dtype='int32'), np.array(pos1_mat, dtype='int32'), np.array(pos2_mat, dtype='int32')


def getWordIdx(token, word2Idx): 
    '''
    Returns from the word2Idx table the word index for a given token
    '''      
    if token in word2Idx:
        return word2Idx[token]
    elif token.lower() in word2Idx:
        return word2Idx[token.lower()]
    else:
        return word2Idx["UNKNOWN_TOKEN"]

def mapDist(dist, dist_map, min_dist=-30, max_dist=30):
    '''
    Map distance
    '''
    if dist in dist_map.keys():
        return dist_map[dist]
    elif dist <= min_dist:
        return dist_map[min_dist]
    elif dist >= max_dist:
        return dist_map[max_dist]
    else:
        raise ValueError('Distance map should cover distance between minimum distance and maximum distance.')


####################################   read training and testing data   #####################################
outputFilePath = 'data/sem-relations.pkl.gz'
embeddingsPath = 'data/embeddings/wiki_extvec.gz' # download English word embeddings from here https://www.cs.york.ac.uk/nlp/extvec/ or https://www.cs.york.ac.uk/nlp/extvec/wiki_extvec.gz
files = ['data/train.txt', 'data/test.txt']
words = {}
max_sent_len = [0,0]

for fileIdx in range(len(files)):
    file = files[fileIdx]
    for line in open(file):
        splits = line.strip().split('\t')       
        label, tokens = splits[0], splits[3].split(' ')
        max_sent_len[fileIdx] = max(max_sent_len[fileIdx], len(tokens))
        for token in tokens:
            words[token.lower()] = True

print("Max Sentence Lengths - traing: {}, testing: {}".format(max_sent_len[0], max_sent_len[1]))
print("Total number of words appears: ", len(words))


################################  word embedding  ###################################
# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []

# :: Load the pre-trained embeddings file ::
fEmbeddings = gzip.open(embeddingsPath, "r") if embeddingsPath.endswith('.gz') else open(embeddingsPath, encoding="utf8")
print("Loading pre-trained embeddings file ...")
for line in fEmbeddings:
    line_split = line.decode('utf-8').strip().split(' ')
    word = line_split[0]
    
    if len(word2Idx) == 0: # Add padding + unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        wordEmbeddings.append(np.zeros(len(line_split)-1)) # Zero vector vor 'PADDING' word
        
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        wordEmbeddings.append(np.random.uniform(-0.25, 0.25, len(line_split)-1))

    # select words appears in training and testing data
    if word.lower() in words:
        wordEmbeddings.append(np.array([float(num) for num in line_split[1:]]))
        word2Idx[word] = len(word2Idx)
          
wordEmbeddings = np.array(wordEmbeddings)
print("Embeddings shape: ", wordEmbeddings.shape)


# :: Create token matrix ::
train_set = createMatrices(files[0], word2Idx, max(max_sent_len))
test_set = createMatrices(files[1], word2Idx, max(max_sent_len))

data = {'wordEmbeddings': wordEmbeddings, 
        'word2Idx': word2Idx, 
        'train_set': train_set,
        'test_set': test_set}
f = gzip.open(outputFilePath, 'wb')
pkl.dump(data, f)
f.close()
print("Data stored in data folder")

        
        