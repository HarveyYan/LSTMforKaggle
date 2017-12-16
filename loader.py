import csv
import nltk
import string
from nltk.corpus import wordnet
import pickle
from keras.preprocessing.text import Tokenizer
import os

MAX_NB_WORDS = 20000
class Data:

    lemmatizer = nltk.WordNetLemmatizer()

    reference = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }

    def __init__(self, id, question1, question2):
        self.id = id
        self.question1 = question1
        self.question2 = question2

    def process(self):
        for question in [self.question1, self.question2]:
            q = nltk.pos_tag(nltk.wordpunct_tokenize(question))
            l = list()
            for token, tag in q:
                if token in string.punctuation:
                    continue
                token = token.lower()
                lemma = self.lemmatizer.lemmatize(token, self.reference.get(tag[0],wordnet.NOUN))
                l.append(lemma)
            if question == self.question1:
                self.question1 = l
            else:
                self.question2 = l

    @classmethod
    def loader(cls, filename, type='train'):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                if type == 'train':
                    temp = TrainData(row[0], row[1], row[2], row[3], row[4], row[5])
                else:
                    temp = TestData(row[0], row[1], row[2])
                temp.process()
                yield temp

    @classmethod
    def get_tokenizer_and_data(cls):
        if not os.path.isfile('cache/texts_train_1'):
            texts_train_1 = list()
            texts_train_2 = list()
            train_labels = list()
            with open('train.csv', 'r') as train_csv:
                reader = csv.reader(train_csv)
                next(reader)
                for row in reader:
                    temp = TrainData(row[0], row[1], row[2], row[3], row[4], row[5])
                    temp.process()
                    texts_train_1.append(temp.question1)
                    texts_train_2.append(temp.question2)
                    # note converting string to integer
                    train_labels.append(int(temp.is_duplicate))
            cls.save('cache/texts_train_1',texts_train_1)
            cls.save('cache/texts_train_2', texts_train_2)
            cls.save('cache/train_labels', train_labels)
        else:
            texts_train_1 = pickle.load(open('cache/texts_train_1','rb'))
            texts_train_2 = pickle.load(open('cache/texts_train_2','rb'))
            train_labels = pickle.load(open('cache/train_labels','rb'))

        if not os.path.isfile('cache/texts_test_1'):
            texts_test_1 = list()
            texts_test_2 = list()
            test_ids = list()
            with open('test.csv', 'r') as test_csv:
                reader = csv.reader(test_csv)
                next(reader)
                for row in reader:
                    temp = TestData(row[0], row[1], row[2])
                    temp.process()
                    texts_test_1.append(temp.question1)
                    texts_test_2.append(temp.question2)
                    test_ids.append(temp.id)
            cls.save('cache/texts_test_1', texts_test_1)
            cls.save('cache/texts_test_2', texts_test_2)
            cls.save('cache/test_ids', test_ids)
        else:
            texts_test_1 = pickle.load(open('cache/texts_test_1','rb'))
            texts_test_2 = pickle.load(open('cache/texts_test_2','rb'))
            test_ids = pickle.load(open('cache/test_ids','rb'))
        print('data processed and loaded')
        train_labels = [int(label) for label in train_labels]
        texts_train_1 = [' '.join(sen) for sen in texts_train_1]
        texts_train_2 = [' '.join(sen) for sen in texts_train_2]
        texts_test_1 = [' '.join(sen) for sen in texts_test_1]
        texts_test_2 = [' '.join(sen) for sen in texts_test_2]
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts_train_1 + texts_train_2 + texts_test_1 + texts_test_2)
        return tokenizer, texts_train_1, texts_train_2, train_labels, texts_test_1, texts_test_2, test_ids

    @classmethod
    def get_training_data(cls):
        if not os.path.isfile('cache/texts_train_1'):
            texts_train_1 = list()
            texts_train_2 = list()
            train_labels = list()
            with open('train.csv', 'r') as train_csv:
                reader = csv.reader(train_csv)
                next(reader)
                for row in reader:
                    temp = TrainData(row[0], row[1], row[2], row[3], row[4], row[5])
                    temp.process()
                    texts_train_1.append(temp.question1)
                    texts_train_2.append(temp.question2)
                    # note converting string to integer
                    train_labels.append(int(temp.is_duplicate))
            cls.save('cache/texts_train_1',texts_train_1)
            cls.save('cache/texts_train_2', texts_train_2)
            cls.save('cache/train_labels', train_labels)
        else:
            texts_train_1 = pickle.load(open('cache/texts_train_1','rb'))
            texts_train_2 = pickle.load(open('cache/texts_train_2','rb'))
            train_labels = pickle.load(open('cache/train_labels','rb'))

        print('training data processed and loaded')
        train_labels = [int(label) for label in train_labels]
        texts_train_1 = [' '.join(sen) for sen in texts_train_1]
        texts_train_2 = [' '.join(sen) for sen in texts_train_2]
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts_train_1 + texts_train_2)
        return tokenizer, texts_train_1, texts_train_2, train_labels

    @classmethod
    def save(cls, filename, data):
        with open(filename, "wb") as output:
            pickle.dump(data, output)

class TrainData(Data):
    def __init__(self,id, qid1, qid2, question1, question2, is_duplicate):
        super(TrainData,self).__init__(id, question1, question2)
        self.qid1 = qid1
        self.qid2 = qid2
        self.is_duplicate = is_duplicate

class TestData(Data):
    def __init__(self, id, question1, question2):
        super(TestData, self).__init__(id, question1, question2)

