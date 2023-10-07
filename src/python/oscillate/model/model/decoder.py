import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D

from oscillate.model.layers.ffn import FeedForwardNetwork


class Decoder(nn.Module):
	def __init__(self, emb_size, block_size, num_heads, ff_size, dtype=torch.float32):
		super().__init__()
		self.emb_size = emb_size
		self.dtype = dtype
		self.pos_encoding = PositionalEncoding1D(emb_size)
		self.self_attn_layer_norm = nn.LayerNorm([block_size, emb_size], dtype=dtype)
		self.enc_attn_layer_norm = nn.LayerNorm([block_size, emb_size], dtype=dtype)
		self.ff_layer_norm = nn.LayerNorm([block_size, emb_size], dtype=dtype)
		self.self_attention = nn.MultiheadAttention(emb_size, num_heads, batch_first=True, dtype=dtype)
		self.encoder_attention = nn.MultiheadAttention(emb_size, num_heads, batch_first=True, dtype=dtype)
		self.ffn = FeedForwardNetwork(emb_size, ff_size, dtype=dtype)

	def forward(self, X, enc_src, enc_pad_mask):
		X = X + self.pos_encoding(X).type(self.dtype)
		attn_out, attn_weights = self.self_attention(X, X, X)
		X = self.self_attn_layer_norm(X + attn_out)

		attn_out, attn_weights = self.encoder_attention(enc_src, enc_src, X, key_padding_mask=enc_pad_mask)
		X = self.enc_attn_layer_norm(X + attn_out)

		ffn_out = self.ffn(X)
		X = self.ff_layer_norm(X + ffn_out)

		return X
