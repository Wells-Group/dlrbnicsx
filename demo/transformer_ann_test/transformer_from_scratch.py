import torch
import numpy as np
import math

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

word_to_ix = {"hello": 0, "world": 1}
embed_instance = Embedding(2, 10)
lookup_tensor = torch.tensor([word_to_ix["world"]], dtype=torch.long)
hello_embed = embed_instance(lookup_tensor)
print(hello_embed)

for param in embed_instance.parameters():
    print(param)


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                # TODO Remove limitation only for d_model % 2 == 0 (even)
                pe[pos, i] = \
                    math.sin(pos/ (10000 ** ((2 * i) / d_model)))
                pe[pos, i+1] = \
                    math.cos(pos/ (10000 ** ((2 * (i + 1)) / d_model)))
                # TODO Why math instead of torch or np
        
        # pe = pe.unsqueeze(0)
        # NOTE register buffer --> requires_grad=False and tensor pe on same device as PositionalEncoder
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        # TODO Why make embeddings relatively larger?
        x = x * math.sqrt(self.d_model)
        # TODO Add constant to embedding
        seq_len = x.size(1)
        x += self.pe[:, :seq_len]
        # NOTE pe not hardcoded to be on cuda
        return x

posEncod = PositionalEncoder(10, 7)
print(posEncod.pe)
print(posEncod.pe.requires_grad)
print(posEncod.pe.device)
posEncod.cuda()
print(posEncod.pe.device)

batch = next(iter(train_iter))
input_seq = batch.English.transpose(0, 1)
input_pad = EN_TEXT.vocab.stoi["pad"]

input_msk = (input_seq != input_pad).unsqueeze(1)

