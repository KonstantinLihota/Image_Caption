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
        self.init_h = nn.Sequential(# cлой для инициализации hidden state rnn сети
            nn.Linear(cnn_feature_size, hidden_dim),
            self.dropout,
            self.relu)
        self.init_c = nn.Sequential(# cлой для инициализации с state rnn сети
            nn.Linear(cnn_feature_size, hidden_dim),

            self.dropout,
            self.relu)

        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim)

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)

        self.n_layers = n_layers

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(2 * hidden_dim, vocab_size)
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
        batch_size = image_vectors.size(0)  # размер батча

        image_vectors = image_vectors.to(device)

        captions_ix = captions_ix.to(device)

        hidden, c = self.init_hidden_state(image_vectors)

        if self.bidirectional == True: # в случае bidirectional == True используется выход модели
            count = 2
        else:
            count = 1

        hidden = hidden.unsqueeze(0).repeat(count * self.n_layers, 1, 1)
        c = c.unsqueeze(0).repeat(count * self.n_layers, 1, 1)

        output = torch.zeros(captions_ix.shape[1], captions_ix.shape[2],# [Count_sent,N_word, batch_size, 2*embeding_dim]
                             batch_size, count * self.hidden_dim).to(device)

        captions_ix = captions_ix.permute(2, 1, 0)

        hid_ix = hidden.unsqueeze(0).repeat(captions_ix.shape[1], 1, 1, 1)  # [Count_sent,N_layers*ciunt, batch_size, embeding_dim]
        c_ix = c.unsqueeze(0).repeat(captions_ix.shape[1], 1, 1, 1)  # [Count_sent,N_layers*ciunt, batch_size, embeding_dim]

        sent_emb = self.embedding(captions_ix) # получаем эмбединги предложений
        for word in range(len(captions_ix)):

            sent = sent_emb[:word + 1]  # [Count_word,Count_sent, batch_size, embeding_dim]

            out = torch.tensor([]).to(device)

            sent = sent.permute(1, 0, 2, 3)
            for captions in range(len(sent)):  # sent - [Cunt_sent, N_word, batch_size, embeding_dim]

                _, (hidden, c) = self.rnn(sent[captions], (hid_ix[captions], c_ix[captions]))

                packed_output = torch.cat((hidden[-1], hidden[-2]), dim=1)

                hid_ix[captions] = hidden
                c_ix[captions] = c

                if out.size(0) == 0:
                    out = packed_output.unsqueeze(0)
                else:

                    out = torch.cat((out, packed_output.unsqueeze(0)), dim=0)

            output[:, word] = out


        logits = self.fc(torch.tensor(output))  # посмотреть размерность hidden

        return logits


