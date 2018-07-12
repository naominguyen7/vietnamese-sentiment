import pandas as pd, numpy as np
import re
import time
import pickle, codecs
from tqdm import tqdm
import logging

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, Bidirectional, LSTM, Dropout, GlobalMaxPool1D
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping

from plot_model_history import plot_model_history
from accent_model import load_accent_model

from sklearn.model_selection import train_test_split

output_dir = 'output/'
log_fname = output_dir + 'log0712.log'
max_features = 12000
maxlen = 300
epochs = 25
batch_size = 64


tokenizer_fname = output_dir + 'tokenizer_{}.pkl'.format(max_features)
model_fname = output_dir + 'mdl0712.h5'
fig_fname = output_dir + 'fig0712.png'

logging.basicConfig(filename=log_fname, level=logging.INFO)
logging.info('Max num of first tokens: {}'.format(max_features))
logging.info('Max length: {}'.format(maxlen))
logging.info('epochs: {}'.format(epochs))
logging.info('batch size: {}'.format(batch_size))
logging.info('File tokenizer: ' + tokenizer_fname)
logging.info('File saved model: ' + model_fname)
logging.info('File saved fig: '+ fig_fname)

neg = pd.read_csv('../data/Negative_train.csv', header=None, names=['cmt'])
neg['sent'] = 'negative'

neu = pd.read_csv('../data/Neutral_train.csv', header=None, names=['cmt'])
neu['sent'] = 'neutral'

pos = pd.read_csv('../data/Positive_train.csv', header=None, names=['cmt'])
pos['sent'] = 'positive'

data = pd.concat((neg, neu, pos))

mycompile = lambda pat:  re.compile(pat,  re.UNICODE)


smiley = mycompile(r'[:=][ \'-]{0,1}[\)dvp]|\^[-_\.o0]?\^')
sad = mycompile(r'[:=][\'-]{0,1}\(|T[-_\.~o0]{0,1}T|=[-_\.~o0"]=')
url_RE = mycompile(r"([https://|http://]+[a-zA-Z\d\/]+[\.]+[a-zA-Z\d\/\.]+)")
ko_pat = mycompile('( ko )|( k )|( kô )|(^k )|(^kô )|(^ko )')
alpha = mycompile(u'[^\w\s]+|\w{10,}')
reduce = mycompile(r'(\w)\1+')

accent_model = load_accent_model('models/v2')
def process_text(text):
    t = url_RE.sub('', text)
    t = t.lower()
    t = reduce.sub(r'\1', t)
    t = smiley.sub(' vui ', t)
    t = sad.sub(' thiu ', t)
    t = alpha.sub('', t)
    if not t:
        return 'neutral'
    if all(ord(c)<128 for c in t):
        t = ' '.join(w for w in t.split() if len(w)<10)
        return ko_pat.sub(u' không ', accent_model.add_accent(t))
    else:
        return ko_pat.sub(u' không ', t)

data['processed'] = data.cmt.apply(process_text)

tokenized_sents = data.processed[~data.processed.isnull()]

tokenizer = Tokenizer(num_words=max_features, lower=False)
tokenizer.fit_on_texts(tokenized_sents)

logging.info('loading word embeddings...')
embeddings_index = {}
f = codecs.open('../wiki.vi.vec', encoding='utf-8')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

logging.info('........densify word_index')
new_word_index = {}
for word, i in tokenizer.word_index.items():
    if i < max_features:
        if word in embeddings_index:
            new_word_index[i] = word

for i in range(1, max_features):
    if i not in new_word_index:
        j = i + 1
        while (j not in new_word_index) & (j < max_features):
            j += 1
        if j < max_features:
            new_word_index[i] = new_word_index.pop(j)

tokenizer.word_index = {w: i for i, w in new_word_index.items()}

len_vocab_tokenized = max(v for v in new_word_index.keys())
logging.info('Done with retokenizing. Reduced to {}'.format(len_vocab_tokenized))
logging.info('Writing tokenizer file')
pickle.dump(tokenizer, open(tokenizer_fname, 'wb'))
# tokenizer = pickle.load(open(tokenizer_fname, 'rb'))

num_words = len(tokenizer.word_index)

embedding_matrix = np.random.normal(0, 1, (num_words+1, 300))


embeddedCount = 0
scale_embedding = False
scale = 1
for word, i in tokenizer.word_index.items():
    embedding_vec = embeddings_index.get(word)
    embedding_matrix[i] = embedding_vec/np.linalg.norm(embedding_vec)

logging.info('Done - Constructed embedding matrix')

# data['len'] = data.processed.apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
# used_data = data[(~data.processed.isnull())&(data.len<7000)]
used_data = data[(~data.processed.isnull())]

# Xtrain = data.processed[~data.processed.isnull()]
# ytrain = data.sent[~data.processed.isnull()]
Xtrain, Xtest, ytrain, ytest = train_test_split(used_data.processed, used_data.sent, test_size=0.2,
                                                stratify=used_data.sent, random_state=7)
tokenized_train = tokenizer.texts_to_sequences(Xtrain)
padded_train = pad_sequences(tokenized_train, maxlen)
ytrain = pd.get_dummies(ytrain)

tokenized_test = tokenizer.texts_to_sequences(Xtest)
padded_test = pad_sequences(tokenized_test, maxlen)
ytest = pd.get_dummies(ytest)

inp = Input(shape=(maxlen, ))
x = Embedding(embedding_matrix.shape[0],
              embedding_matrix.shape[1],
              weights=[embedding_matrix], trainable=False
             )(inp)
x = Bidirectional(LSTM(4, return_sequences=True,
                       name='lstm_layer',
                       dropout=0.5, recurrent_dropout=0.5
                      ))(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.2)(x)
x = Dense(16, activation='relu')(x)
x = Dense(3, activation='sigmoid')(x)
model = Model(input=inp, outputs=x)

opt = optimizers.adam(lr=0.001)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

early_stop = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.0001, patience=2)

start = time.time()
model_info = model.fit(padded_train, ytrain, validation_data=(padded_test, ytest),
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=[early_stop],
                       verbose=1
                       )
logging.info('Training took {}'.format(time.time()-start))


model.save(model_fname)
logging.info(model_info.history.keys())
plot_model_history(model_info, fig_fname)
logging.info('Performance on test: {}'.format(model.evaluate(padded_test,ytest)))


