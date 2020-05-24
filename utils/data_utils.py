import os
import numpy as np
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import spacy

def read_data():
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_md')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def build_embedding(pretrained_model, field):
        embedding = [np.random.rand(1, 300) for _ in range(4)]
        for word in field.vocab.stoi:
            if word in {'<unk>', '<pad>', '<sos>', '<eos>'}:
                continue
            if isinstance(pretrained_model, spacy.lang.en.English):
                embedding.append(pretrained_model.vocab[word].vector.reshape(1, 300))
            else:
                embedding.append(pretrained_model.wv[word].reshape(1, 300))
        return np.concatenate(embedding)

    SRC = Field(tokenize=tokenize_de, init_token='GO', eos_token='EOS', lower=True)
    TRG = Field(tokenize=tokenize_en, init_token='GO', eos_token='EOS', lower=True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
    image_train_data = np.zeros(len(train_data), 48, 48, 3)
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    de_w2v_path = 'w2v_models/de_w2v_300d.model'
    if os.path.exists(de_w2v_path):
        gensim_de = Word2Vec.load(de_w2v_path)
    else:
        gensim_de = Word2Vec([s for s in train_data.src], size=300, window=5, min_count=1, workers=8)
        gensim_de.save(de_w2v_path)

    encoder_embedding_matrix = build_embedding(gensim_de, SRC)
    decoder_embedding_matrix = build_embedding(spacy_en, TRG)

    return SRC, TRG, train_data, image_train_data, valid_data, test_data, encoder_embedding_matrix, decoder_embedding_matrix

def tokenize_sequence(sentences, filters, max_num_words, max_vocab_size):
    """
    Tokenizes a given input sequence of words.

    Args:
        sentences: List of sentences
        filters: List of filters/punctuations to omit (for Keras tokenizer)
        max_num_words: Number of words to be considered in the fixed length sequence
        max_vocab_size: Number of most frequently occurring words to be kept in the vocabulary

    Returns:
        x : List of padded/truncated indices created from list of sentences
        word_index: dictionary storing the word-to-index correspondence

    """

    sentences = [' '.join(word_tokenize(s)[:max_num_words]) for s in sentences]

    tokenizer = Tokenizer(filters=filters)
    tokenizer.fit_on_texts(sentences)

    word_index = dict()
    word_index['PAD'] = 0
    word_index['UNK'] = 1
    word_index['GO'] = 2
    word_index['EOS'] = 3

    for i, word in enumerate(dict(tokenizer.word_index).keys()):
        word_index[word] = i + 4

    tokenizer.word_index = word_index
    x = tokenizer.texts_to_sequences(list(sentences))

    for i, seq in enumerate(x):
        if any(t >= max_vocab_size for t in seq):
            seq = [t if t < max_vocab_size else word_index['UNK'] for t in seq]
        seq.append(word_index['EOS'])
        x[i] = seq

    x = pad_sequences(x, padding='post', truncating='post', maxlen=max_num_words, value=word_index['PAD'])

    word_index = {k: v for k, v in word_index.items() if v < max_vocab_size}

    return x, word_index


def create_embedding_matrix(word_index, embedding_dim, w2v_path):
    """
    Create the initial embedding matrix for TF Graph.

    Args:
        word_index: dictionary storing the word-to-index correspondence
        embedding_dim: word2vec dimension
        w2v_path: file path to the w2v pickle file

    Returns:
        embeddings_matrix : numpy 2d-array with word vectors

    """
    w2v_model = gensim.models.Word2Vec.load(w2v_path)
    embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(word_index), embedding_dim))
    for word, i in word_index.items():
        try:
            embeddings_vector = w2v_model[word]
            embeddings_matrix[i] = embeddings_vector
        except KeyError:
            pass

    return embeddings_matrix


def create_data_split(x, img_x, y):
    """
    Create test-train split according to previously defined CSV files
    Depending on the experiment - qgen or dialogue

    Args:
        x: input sequence of indices
        img_x: input image
        y: output sequence of indices
        experiment: dialogue (conversation system) or qgen (question generation) task

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test: train val test split arrays

    """

    train_size, val_size, test_size = 29000, 1013, 1000 # Multi30k
    train_indices = range(train_size)
    val_indices = range(train_size, train_size + val_size)
    test_indices = range(train_size + val_size, train_size + val_size + test_size)

    x_train = x[train_indices]
    img_x_train = img_x[train_indices]
    y_train = y[train_indices]
    x_val = x[val_indices]
    img_x_val = img_x[train_indices]
    y_val = y[val_indices]
    x_test = x[test_indices]
    img_x_test = img_x[train_indices]
    y_test = y[test_indices]

    return x_train, img_x_train, y_train, x_val, img_x_val, y_val, x_test, img_x_test, y_test


def get_batches(x, x_img, y, batch_size):
    """
    Generate inputs and targets in a batch-wise fashion for feed-dict

    Args:
        x: entire source sequence array
        y: entire output sequence array
        batch_size: batch size

    Returns:
        x_batch, y_batch, source_sentence_length, target_sentence_length

    """

    for batch_i in range(0, len(x) // batch_size):
        start_i = batch_i * batch_size
        x_batch = x[start_i: start_i + batch_size]
        img_x_batch = x_img[start_i: start_i + batch_size]
        y_batch = y[start_i: start_i + batch_size]

        source_sentence_length = [np.count_nonzero(seq) for seq in x_batch]
        target_sentence_length = [np.count_nonzero(seq) for seq in y_batch]

        yield x_batch, img_x_batch, y_batch, source_sentence_length, target_sentence_length


def build_image_mask(self, batch_size, image_size):
    return np.zeros([batch_size, image_size, image_size, 3])
