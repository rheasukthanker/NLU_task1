import numpy as np
from data_handler import DataHandler
from nltk.probability import FreqDist
from tags import NUM_TAGS, TAGS, Tag


class VocabHandler:
    def __init__(self, data_handler, vocab_size):
        fdist = FreqDist()
        tag_tokens = TAGS.values()
        for sent in data_handler.get_corpus():
            for word in sent:
                if word not in tag_tokens:
                    fdist[word] += 1

        vocab_words = fdist.most_common(vocab_size - len(TAGS))
        self.vocab = {}
        i = 0
        for word, _ in vocab_words:
            self.vocab[word] = i
            i += 1
        for tag in TAGS.values():
            self.vocab[tag] = i
            i += 1
        assert(i <= vocab_size)
        print("Defined vocabulary with " + str(len(self.vocab)) + " words. (including tags)")

        self.inv_vocab = {}
        for key, val in self.vocab.items():
            self.inv_vocab[val] = key


    def vocab_indices(self, sentence_batch):
        batch_size, sent_len = len(sentence_batch), len(sentence_batch[0])
        indices = np.empty((batch_size, sent_len), dtype=np.int32)
        for i, sent in enumerate(sentence_batch):
            for j, word in enumerate(sent):
                if word in self.vocab:
                    indices[i, j] = self.vocab[word]
                else:
                    indices[i, j] = self.vocab[TAGS[Tag.UNK]]

        return indices


    def vocab_words(self, index_batch, sent_lens_without_tags):
        batch_size = len(index_batch)
        sentences = []
        for indices, curr_len in zip(index_batch, sent_lens_without_tags):
            word_list = []
            for i in range(curr_len + 1):
                word_list.append(self.inv_vocab[indices[i]])
            sentences.append(word_list)
        return sentences

