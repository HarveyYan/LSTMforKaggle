from loader import Data
import gensim
import numpy as np
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from keras.utils import plot_model
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc

'''
hyper parameters
'''
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 30
BATCH_SIZE = 10
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

'''
optimizable hyperparams
'''
num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25
act = 'relu'
re_weight = True
glove = False

embedding_mat = None
length = None
STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)
outpath = './result/' + STAMP + '/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

'''
load pretrained word embeddings, whether it's word2vec or glove
'''
def load_word2vec():
    model = gensim.models.KeyedVectors.load_word2vec_format('word_embedding/GoogleNews-vectors-negative300.bin.gz', binary=True)
    return model

def load_glove():
    model = {}
    f = open('word_embedding/glove.840B.300d.txt')
    for line in f:
        values = line.split()
        model[values[0]] = np.asarray(values[1:], dtype='float32')
    f.close()
    return model

'''
embedding matrix for embedding layer
'''
def generate_embeddings(model, tokenizer):
    global embedding_mat, length
    missing_vocab = open(outpath + 'missing_vocab.txt', 'w')
    word_index = tokenizer.word_index
    print('total words found:', len(word_index))
    length = min(MAX_NB_WORDS, len(word_index)) + 1
    embedding_mat = np.zeros((length, EMBEDDING_DIM))

    if glove:
        for word, i in word_index.items():
            embedding_vector = model.get(word)
            if embedding_vector is None:
                print(word, 'not in model vocab')
                missing_vocab.write(word + '\n')
            else:
                embedding_mat[i] = embedding_vector
    else:
        for word, i in word_index.items():
            if i > length:
                continue
            if word in model.vocab:
                embedding_mat[i] = model.word_vec(word)
            else:
                print(word, 'not in model vocab')
                # missing_vocab.write(word+'\n')
    missing_vocab.close()



def plot_some_curves(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(outpath +'model_acc.png')
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(outpath + 'model_loss.png')
    plt.close()


'''
obsolete
'''
def get_a_batch(generator, tokenizer):
    batch = []
    for i in range(BATCH_SIZE):
        for sample in generator:
            q1 = tokenizer.texts_to_sequences(sample.question1)
            q2 = tokenizer.texts_to_sequences(sample.question2)
            batch.append([pad_sequences(q1, maxlen=MAX_SEQUENCE_LENGTH)
                             , pad_sequences(q2, maxlen=MAX_SEQUENCE_LENGTH)
                             , sample.is_duplicate])
    return batch

class LSTM_model:

    class_weights = {0:1.309028344, 1:0.472001959}
    '''
    We know that there are 36.92% positive entities in the train set and about
     17.46% positive entities in the test set, so in order to map the share of
      positive entities to be the same, one positive entity in the train set
      counts for 0.1746 / 0.3692 = 0.472 positive entities in the test set.
    Similarly, the weight of negative entities in the train set is (1 - 0.1746) / (1 - 0.3692) = 1.309 .
    '''
    def __init__(self, re_weight = False):
        self.re_weight = re_weight

        embedding_layer = Embedding(
            length,
            EMBEDDING_DIM,
            weights=[embedding_mat],
            input_length = MAX_SEQUENCE_LENGTH,
            trainable = False)

        lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

        sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        x1 = lstm_layer(embedded_sequences_1)

        sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        y1 = lstm_layer(embedded_sequences_2)

        merged = concatenate([x1, y1])
        normzalized_merged = BatchNormalization()(Dropout(rate_drop_dense)(merged))

        dense = Dense(num_dense, activation='relu')(normzalized_merged)
        normalized_dense = BatchNormalization()(Dropout(rate_drop_dense)(dense))

        preds = Dense(1, activation='sigmoid')(normalized_dense)

        self.net = Model(inputs=[sequence_1_input, sequence_2_input], outputs= preds)
        self.net.compile(
            loss='binary_crossentropy',
            optimizer='nadam',
            metrics=['acc']
        )

    def train(self,train_1, train_2, labels):
        indices = np.random.permutation(len(train_1))
        idx_train = indices[:int(len(train_1)*(1-VALIDATION_SPLIT))]
        idx_val = indices[int(len(train_1) * (1 - VALIDATION_SPLIT)):]

        data_1_train = np.vstack((train_1[idx_train],train_2[idx_train]))
        data_2_train = np.vstack((train_2[idx_train], train_1[idx_train]))
        labels_train = np.concatenate((labels[idx_train],labels[idx_train]))

        data_1_val = np.vstack((train_1[idx_val], train_2[idx_val]))
        data_2_val = np.vstack((train_2[idx_val], train_1[idx_val]))
        labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

        # Must investigate this part further
        weight_val = np.ones(len(labels_val))
        if re_weight:
            weight_val *= 0.472001959
            weight_val[labels_val == 0] = 1.309028344

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        best_model_path = outpath + 'weight.h5'
        model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
        hist = self.net.fit([data_1_train, data_2_train], labels_train,
                     validation_data=([data_1_val,data_2_val],labels_val,weight_val),
                     epochs=200, batch_size=2048,shuffle=True,
                     class_weight=self.class_weights, callbacks=[early_stopping, model_checkpoint])

        plot_model(self.net, to_file=outpath+'net.png')
        self.plot_roc([data_1_val,data_2_val], labels_val)
        np.savetxt(outpath + 'acc.txt', np.array(hist.history['acc']), delimiter=',')
        np.savetxt(outpath + 'val_acc.txt', np.array(hist.history['val_acc']), delimiter=',')
        np.savetxt(outpath + 'loss.txt', np.array(hist.history['loss']), delimiter=',')
        np.savetxt(outpath + 'val_loss.txt', np.array(hist.history['val_loss']), delimiter=',')
        plot_some_curves(hist)

        self.net.load_weights(best_model_path)

    def predict(self, test_1, test_2, ids, best_val_score):
        preds = self.net.predict([test_1, test_2], batch_size=8192, verbose=1)
        preds +=self.net.predict([test_2, test_1], batch_size=8192, verbose=1)
        preds /= 2
        submission = pd.DataFrame({'test_id': ids, 'is_duplicate': preds.ravel()})
        submission.to_csv(outpath + 'prediction.csv', index=False)

    def plot_roc(self, data_validation, labels_validation):
        preds = self.net.predict(data_validation, batch_size=2048)
        fpr, tpr, _ = roc_curve(labels_validation, preds)
        roc_auc = auc(fpr,tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig(outpath+'roc.png')
        plt.close()

'''
todo, process training and testing data
'''
if __name__ == "__main__":
    tokenizer, texts_train_1, texts_train_2, train_labels, texts_test_1, texts_test_2, test_ids = Data.get_tokenizer_and_data()
    print('padding and converting sequences')
    train_1 = pad_sequences(tokenizer.texts_to_sequences(texts_train_1), maxlen=MAX_SEQUENCE_LENGTH)
    train_2 = pad_sequences(tokenizer.texts_to_sequences(texts_train_1), maxlen=MAX_SEQUENCE_LENGTH)
    test_1 = pad_sequences(tokenizer.texts_to_sequences(texts_test_1), maxlen=MAX_SEQUENCE_LENGTH)
    test_2 = pad_sequences(tokenizer.texts_to_sequences(texts_test_2), maxlen=MAX_SEQUENCE_LENGTH)
    labels = np.array(train_labels)
    print('acquired training and testing data')

    if glove:
        if os.path.exists('word_embedding/glove.840B.300d.txt'):
            model = load_glove()
        else:
            raise RuntimeError('glove word embedding doesn\'t exist')
    else:
        if os.path.exists('word_embedding/GoogleNews-vectors-negative300.bin.gz'):
            model = load_word2vec()
        else:
            raise RuntimeError('google word2vec word embedding doesn\'t exist')
    generate_embeddings(model, tokenizer)
    print('acquired embedding matrix')

    lstm = LSTM_model(re_weight=True)
    print('LSTM assembled')
    best_val_score = lstm.train(train_1, train_2, labels)
    # lstm.predict(test_1, test_2, test_ids, best_val_score)