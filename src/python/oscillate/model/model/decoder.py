import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D

from oscillate.model.layers.ffn import FeedForwardNetwork


class Decoder(nn.Module):
	def __init__(self, emb_size, block_size, num_heads, ff_size):
		super().__init__()
		self.emb_size = emb_size
		self.pos_encoding = PositionalEncoding1D(emb_size)
		self.self_attn_layer_norm = nn.LayerNorm([block_size, emb_size])
		self.enc_attn_layer_norm = nn.LayerNorm([block_size, emb_size])
		self.ff_layer_norm = nn.LayerNorm([block_size, emb_size])
		self.self_attention = nn.MultiheadAttention(emb_size, num_heads, batch_first=True)
		self.encoder_attention = nn.MultiheadAttention(emb_size, num_heads, batch_first=True)
		self.ffn = FeedForwardNetwork(emb_size, ff_size)

	def forward(self, X, enc_src, enc_pad_mask):
		X = X + self.pos_encoding(X)
		attn_out, attn_weights = self.self_attention(X, X, X)
		X = self.self_attn_layer_norm(X + attn_out)

		attn_out, attn_weights = self.encoder_attention(enc_src, enc_src, X, key_padding_mask=enc_pad_mask)
		X = self.enc_attn_layer_norm(X + attn_out)

		ffn_out = self.ffn(X)
		X = self.ff_layer_norm(X + ffn_out)

		return X
