import torch, torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CaptionNet(nn.Module):
    def __init__(self, hidden_dim, cnn_feature_size=2048, dropout=0.1, n_layers=2, bidirectional=True,
                 embedding_dim=250, vocab_size=tokenizer.vocab_size):
        super(self.__class__, self).__init__()
        '''
        hidden_dim: размерность скрытых состояний rnn сети 
        cnn_feature_size: размерность выхода из энкодера картинки
        embedding_dim: размернсть эмбединга 
        n_layers количество слоев в rnn сети
        '''
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.embedding_dim = embedding_dim

        self.init_h = nn.Sequential( # cлой для инициализации hidden state rnn сети
            nn.Linear(cnn_feature_size, hidden_dim * 2),
            self.dropout,
            self.relu,
            nn.Linear(hidden_dim * 2, hidden_dim),
            self.dropout,
            self.relu)

        self.init_c = nn.Sequential(
            nn.Linear(cnn_feature_size, hidden_dim * 2),
            self.dropout,
            self.relu,
            nn.Linear(hidden_dim * 2, hidden_dim),
            self.dropout,
            self.relu)

        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim)

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)

        # self.init_h = nn.Linear(cnn_feature_size, hidden_dim)  # linear layer to find initial hidden state of LSTMCell
        # self.init_c = nn.Linear(cnn_feature_size, hidden_dim)   # linear layer to find initial cell state of LSTMCell
        # self.f_beta = nn.Linear(hidden_dim, cnn_feature_size)  # linear layer to create a sigmoid-activated gate

        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(2 * hidden_dim, vocab_size)  # посчитать длину словаря
        self.init_weights()

    def init_weights(self):
        """
        Инициализация весов из равномерного распределения для обеспечения быстрой сходимости
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Инициализация скрытых состояний
        """

        h = self.init_h(encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(encoder_out)
        return h, c

    def forward(self, image_vectors, captions_ix):
        """
        image_vectors:  эмбединг картинки
        caption_ix:  тензор с несколькими описаниями картинки
        """
        # batch_size = image_vectors.size(0)
        # размер батча
        # vocab_size = self.vocab_size

        image_vectors = image_vectors.to(device)
        captions_ix = captions_ix.to(device)
        hidden, c = self.init_hidden_state(image_vectors) #инициализация скрытых состояний

        hidden = hidden.unsqueeze(0).repeat(captions_ix.size(1) - 1, 1, 1)
        c = c.unsqueeze(0).repeat(captions_ix.size(1) - 1, 1, 1)

        sent = torch.tensor([]).to(device) # sent содержит сгенирированые о
                                           # писания на i том шаге по предложению
        output = torch.tensor([]).to(device)

        for word in captions_ix.permute(2, 1, 0):   # Для каждого слова из описания генерируем следующее слово
            x = self.embedding(word)

            if sent.size() == 0:
                sent = x.unsqueeze(1)
            else:
                sent = torch.cat((sent, x.unsqueeze(1)), dim=1)

            out = torch.tensor([]).to(device)
            hid_ix = [hidden]
            c_ix = [c]
            for captions in range(len(sent)):
                packed_output, (hidden, c) = self.rnn(sent[captions], (hid_ix[captions], c_ix[captions]))
                hid_ix.append(hidden)
                c_ix.append(c)

                if out.size(0) == 0:
                    out = packed_output.unsqueeze(0)
                else:

                    out = torch.cat((out, packed_output.unsqueeze(0)), dim=0)

        # 1. инициализируем LSTM state
        # 2. применим слой эмбеддингов к image_vectors
        # 3. скормим LSTM captions_emb
        # 4. посчитаем логиты из выхода LSTM
        logits = self.fc(torch.tensor(out))

        return logits