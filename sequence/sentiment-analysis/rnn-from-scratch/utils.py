import torch

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f"{round(self.avg, 4)}"

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_vocab(sentences):
    vocab = set()
    for sentence in sentences:
        for word in sentence.split():
            vocab.add(word)
    return sorted(list(vocab))

def generate_training_data(sentences, labels, vocab):
    assert len(sentences) == len(labels)

    x_train = []
    t_train = []

    for i in range(len(sentences)):
        words = sentences[i].split()

        x = torch.zeros(len(words), len(vocab))
        t = torch.zeros(1, 1)

        for j, w in enumerate(words):
            index = vocab.index(w)
            x[j,index] = 1

        t[0,0] = labels[i]

        x_train.append(x)
        t_train.append(t)

    return x_train, t_train


