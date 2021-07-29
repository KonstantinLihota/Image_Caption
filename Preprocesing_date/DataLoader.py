from torch.utils.data import Sampler
import random
from torch.utils.data import DataLoader
import numpy as np

class ReviewsSampler(Sampler): # cоздание сэмплов заданой длинны бытча
    def __init__(self, subset, batch_size=32):
        self.batch_size = batch_size
        self.subset = subset

        self.indices = subset.indices

        self.tokenized = np.array(subset.dataset.tokenized)[self.indices]

    def __iter__(self):

        batch_idx = []
        for index in np.argsort(list(map(len, self.tokenized))):
            batch_idx.append(index)
            if len(batch_idx) == self.batch_size:
                yield batch_idx
                batch_idx = []

        if len(batch_idx) > 0:
            yield batch_idx

    def __len__(self):
        return len(self.dataset)




'''
get_padded Приводит предложение к одинаковой длине max_len = 20,
добавление pad токенов (0)
values: батч состоящий из предложений
'''
def get_padded(values):
    padded = []
    max_len = 20

    for sent in values:
        if len(sent) > max_len:
            padded.append(np.array(sent[:max_len - 1] + [102]))


        else:
            padded.append(np.array(sent + [0] * (max_len - len(sent))))

    return np.array(padded)

# Вспомогательная функция формирования батча
def collate_fn(batch):
    inputs = []
    labels = []
    bh = []
    for elem in batch:
        x = random.randint(0, len(elem['tokenized']) - 1)

        inputs.append(elem['tokenized'])

        labels.append(elem['img_codes'])

    # print(inputs)
    inp = []

    for i in inputs:
        inp.append(get_padded(i))  ## padded inputs

    return {"inputs": torch.tensor(inp), "img_codes": torch.FloatTensor(labels)}
