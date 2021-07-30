from tags import NUM_TAGS, TAGS, Tag
import numpy as np
import random


class DataHandler:
    def __init__(self, path, sentence_len_with_tags, batch_size, shuffle=False, add_eos=True):
        self.sentence_len_with_tags = sentence_len_with_tags
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.add_eos = add_eos
        self.load_corpus(path)
        self.remove_long_sentences()
        self.store_sentence_lengths()
        self.normalize_sentences()

    def get_corpus(self):
        return self.corpus

    def load_corpus(self, path):
        with open(path, 'r') as data_file:
            content = data_file.readlines()
        self.corpus = [sent.strip().split() for sent in content]
        print('Loaded corpus "{}" with {} sentences'.format(path, len(self.corpus)))

    def remove_long_sentences(self):
        trunc_corpus = []
        for i, sent in enumerate(self.corpus):
            if len(sent) + 2 <= self.sentence_len_with_tags:
                trunc_corpus.append(sent)

        del self.corpus[:]
        self.corpus =trunc_corpus
        print("Truncated corpus. Now has " + str(len(self.corpus)) + " sentences.")

    def store_sentence_lengths(self):
        self.sentence_lens_without_tags = np.array([len(sent) for sent in self.corpus], dtype=np.int32)

    def normalize_sentences(self):
        for idx, sent in enumerate(self.corpus):
            num_pad = self.sentence_len_with_tags - (len(sent) + 2)
            if self.add_eos:
                self.corpus[idx] = [TAGS[Tag.BOS]] + sent + [TAGS[Tag.EOS]] + num_pad*[TAGS[Tag.PAD]]
            else:
                self.corpus[idx] = [TAGS[Tag.BOS]] + sent + (num_pad+1)*[TAGS[Tag.PAD]]

        print('Normalized corpus. New length is {}'.format(len(self.corpus)))

    def __iter__(self):
        self.current_batch_number = 0
        if self.shuffle:
            zipped = list(zip(self.corpus, self.sentence_lens_without_tags))
            random.shuffle(zipped)
            self.corpus, self.sentence_lens_without_tags = zip(*zipped)

        return self

    def __next__(self):
        """
        Returns
        -------
        batch_x : np.ndarray with shape (batch_size, self.sentence_len_with_tags)
            batch_x[i, j] contains the jth word of the tagged version of the ith sentence.

        batch_y : np.ndarray with shape (batch_size, self.sentence_len_with_tags - 1)
            batch_y[i, j] contains the (j+1)th word of the tagged version of the ith sentence. As a
            direct consequence, batch_y never contains the Tags.BOS token.

        batch_len : np.ndarray with shape (batch_size,)
            batch_len[i] contains the length of the original (untagged) version of the ith sentence in the batch.
        """
        num_sent = len(self.corpus)
        beg = self.current_batch_number * self.batch_size
        end = min(num_sent, (self.current_batch_number + 1) * self.batch_size)

        if beg >= num_sent:
            raise StopIteration

        batch_x = self.corpus[beg:end]
        batch_y = [sent[1:] for sent in batch_x]
        batch_len = self.sentence_lens_without_tags[beg:end]

        self.current_batch_number += 1
        return batch_x, batch_y, batch_len
