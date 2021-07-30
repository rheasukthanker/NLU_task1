NUM_TAGS = 4
class Tag:
    BOS = 0
    EOS = 1
    PAD = 2
    UNK = 3
TAGS = {
    Tag.BOS: '<bos>',
    Tag.EOS: '<eos>',
    Tag.PAD: '<pad>',
    Tag.UNK: '<unk>',
}
