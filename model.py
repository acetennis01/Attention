import torch
import torch.nn as nn
import math

# converts the sentence into a vector of dimension 512(Embeddings)


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncodings(nn.Module):

    def __init__(self, d_model: int, seq_length: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        # getting the positional infomation

        # matrix of shape seq_len * d_model
        pos_encoding = torch.zeros(seq_length, d_model)

        # vector of shape seq_len
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        denom = torch.exp(torch.arange(0, d_model, 2).float()
                          * (-math.log(10000.0) / d_model))

        pos_encoding[:, 0::2] = torch.sin(position * denom)
        pos_encoding[:, 1::2] = torch.cos(position * denom)

        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer('pe', pos_encoding)

    def forward(self, x):
        # stays fixed, should not change
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

# Add&Norm

class LayerNorm(nn.Module):

    def __init__(self, epsilon: float = 10**-6):
        super().__init__()
        # eplison is for numerical stability
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones(1))  # *
        self.gamma = nn.Parameter(torch.zeros(1))  # +

    # x_proj = (x - mean)/sqrt(std^2 + epsilon)
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.epsilon) + self.gamma

# FFN(x) = max(0, xW1 + b1)W2 + b2

class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)  # W1 b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  # W2 b2

    def forward(self, x):
        #                                linear1                            linear2
        # (batch, dim(seq_length, d_model)) --> (batch, dim(seq_length, d_ff)) --> (batch, dim(seq_length, d_model))
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

# MultiHead(Q, K, V ) = Concat(head1, ..., headh)W_O
#                       where head_i = Attention(QxW_Q_i , KxW_K_i , VxW_V_i )
# Attention(Q, K, V ) = softmax(QK_T/sqrt(d_k))V

class MultiHead(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        # dk = dv = dmodel/h

        self.d_k = d_model // h

        # setting up a matrix with dimensions of d_model x d_model
        self.w_q = nn.Linear(d_model, d_model)  # W_q
        self.w_k = nn.Linear(d_model, d_model)  # W_k
        self.w_v = nn.Linear(d_model, d_model)  # W_v

        self.w_o = nn.Linear(d_model, d_model)  # W_o

        self.dropout = nn.Dropout(dropout)

    # Attention(Q, K, V ) = softmax(QK_T/sqrt(d_k))V
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # QK_T/sqrt(d_k)
        attenstion_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # apply the mask
        if mask is not None:
            attenstion_scores.masked_fill_(mask == 0, -1e9)

        if dropout is not None:
            attenstion_scores = dropout(attenstion_scores)

        # apply the softmax
        # softmax(QK_T/sqrt(d_k))
        attenstion_scores = attenstion_scores.softmax(dim=-1)

        #              softmax(..)xV
        return (attenstion_scores @ value), attenstion_scores

    def forward(self, q, k, v, mask):

        # q' = Q X W_Q : (seq, d_model) x (d_model, d_model) = (seq, d_model)
        query = self.w_q(q)
        # k' = K X W_K : (seq, d_model) x (d_model, d_model) = (seq, d_model)
        key = self.w_k(k)
        # v' = V X V_Q : (seq, d_model) x (d_model, d_model) = (seq, d_model)
        value = self.w_v(v)

        # (batch, seq_length, d_model) --> (batch, seq_length, h, d_k) --> (batch, h, seq_length, d_model)
        query = query.view(
            query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1],
                       self.h, self.d_k).transpose(1, 2)
        value = value.view(
            value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHead.attention(
            query, key, value, mask, self.dropout)

        # (batch, h, seq_length, d_k) --> (batch, seq_length, h, d_k) --> (batch, seq_length, d_model)
        x = x.transpose(1, 2).contiguous().view(
            x.shape[0], -1, self.h * self.d_k)

        # MultiHead(Q, K, V) = Concat(head_i, .., head_h) x W_o
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, droupout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(droupout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHead, ff_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.ff_block = ff_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.ff_block)
        return x

# run the EncoderBlock N times

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        # Take the Layer Normalization of x
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHead, cross_attention_block: MultiHead, ff_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.ff_block = ff_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(
            x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.ff_block)
        return x

# runs the decoder block n times

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers

        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)

        return self.norm(x)

# Linear + Softmax

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_length, d_model) --> (bath, seq_length, vocab_size)
        return torch.softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncodings, tgt_pose: PositionalEncodings, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pose = tgt_pose
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pose(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

def transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Positional Encoding
    src_pos_encod = PositionalEncodings(d_model, src_seq_len, dropout)
    tgt_pos_encod = PositionalEncodings(d_model, tgt_seq_len, dropout)

    # Encoder Blocks
    encoder_blocks = []
    for i in range(N):
        encoder_attention_block = MultiHead(d_model, h, dropout)
        ff_block = FeedForward(d_model, d_ff, dropout)
        encoder_b = EncoderBlock(encoder_attention_block, ff_block, dropout)
        encoder_blocks.append(encoder_b)

    # Decoder Blocks
    decoder_blocks = []
    for i in range(N):
        decoder_self_attention_block = MultiHead(d_model, h, dropout)
        decoder_cross_attention_block = MultiHead(d_model, h, dropout)
        ff_block = FeedForward(d_model, d_ff, dropout)
        decoder_b = DecoderBlock(
            decoder_self_attention_block, decoder_cross_attention_block, ff_block, dropout)
        decoder_blocks.append(decoder_b)

    # encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # transformer
    transformer = Transformer(encoder, decoder, src_embed,
                              tgt_embed, src_pos_encod, tgt_pos_encod, projection_layer)

    # parameter
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return transformer
