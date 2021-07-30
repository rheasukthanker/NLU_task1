import numpy as np


def perplexity(probs, word_indices, sentence_lengths_without_tags):
    """
    Compute perplexity of a given batch of sequence of words.

    Parameters
    ----------
    probs: Tensor of shape [batch_size, sequence_length - 1, vocab_size], containing the
           probability of each word in the vocabulary for indices of the
           given sequence. More formally, probs[i, t, :] must contain the
           distribution

           P(w_t | w_{t-1}, ..., w_0)

           for ith sentence in the batch.

    word_indices: Tensor of shape [batch_size, sequence_length - 1], containing the
           vocabulary indices of the ground truth words.

    sentence_lengths_without_tags: Tensor of shape [batch_size], containing the lengths of
           untagged version of each sentence.

    Returns
    -------
    perplexity: Tensor of shape [batch_size] containing float values equal to
                the perplexity of the corresponding sentence.
    """
    predict_lens = sentence_lengths_without_tags + 1
    batch_size, n = word_indices.shape
    assert(n == probs.shape[1])

    ii = np.repeat(np.arange(batch_size), n)
    jj = np.tile(np.arange(n), batch_size)

    logprobs = np.log(probs[ii, jj, word_indices.flatten()]).reshape(batch_size, n)
    mask = np.arange(n)[None, :] >= predict_lens[:, None]
    logprobs[mask] = 0
    return np.exp(-np.sum(logprobs, axis=1) / predict_lens)
