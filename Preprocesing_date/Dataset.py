from torch.utils.data import Dataset, random_split
import torch
'''
В классе описывается логика формирования датасета
img_codes: картинка(или эбединг картинки)
text должен состоять из нескольких предложений описывающих картинку 
tokenizer: Токенизатор предложения
'''

class PreprocessingDataset(Dataset):

    def __init__(self, text, img_codes, tokenizer):

        self.tokenized = []
        for description in text:
            batch = []
            for sent in description: #проход по каждому из описаний к конкретной картинке
                batch.append(tokenizer(sent)['input_ids'])
            self.tokenized.append(batch)

        self.text = text
        self.img_codes = img_codes

    def __getitem__(self, index):
        return {"tokenized": self.tokenized[:][index], "img_codes": self.img_codes[index]}

    def __len__(self):
        return len(self.text)


