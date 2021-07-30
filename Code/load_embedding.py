from gensim import models
import tensorflow as tf
import numpy as np


def load_embedding(session, vocab, emb, path, dim_embedding, vocab_size):
    """
    :param session:         Tensorflow session object
    :param vocab:           A dictionary mapping token strings to vocabulary IDs
    :param emb:             Embedding tensor of shape vocabulary_size x dim_embedding
    :param path:            Path to embedding file
    :param dim_embedding:   Dimensionality of the external embedding.
    :param vocab_size:      Size of vocabulary
    :return: None
    """

    print("Loading external embeddings from %s" % path)

    model = models.KeyedVectors.load_word2vec_format(path, binary=False)  
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0

    for tok, idx in vocab.items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)
        
    print("%d words out of %d could be loaded" % (matches, vocab_size))
    
    pretrained_embeddings = tf.placeholder(tf.float32, [None, None]) 
    assign_op = emb.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embedding})  # here, embeddings are actually set
