import typing

import torch
import torch.nn as nn

from positional_encodings.torch_encodings import PositionalEncoding1D

from oscillate.model.layers.add_norm import AddNorm
from oscillate.model.layers.ffn import FeedForwardNetwork


class Encoder(nn.Module):

	def __init__(
			self,
			block_size: int,
			emb_size: int,
			mha_heads: int,
			ff_size: int,
			*args,
			pad_token: int = 0,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.block_size = block_size
		self.emb_size = emb_size
		self.pos_encoding = PositionalEncoding1D(emb_size)
		self.mha = nn.MultiheadAttention(emb_size, mha_heads)
		self.add_norm = AddNorm((block_size, emb_size))
		self.ffn = FeedForwardNetwork(emb_size, ff_size)
		self.pad_token = pad_token

	def forward(self, X: torch.Tensor):
		pad_mask = torch.transpose(torch.all(X == torch.zeros((X.shape[2],)), dim=2), 0, 1)
		y = X + self.pos_encoding(X)
		attn_output, attn_weights = self.mha(y, y, y, key_padding_mask=pad_mask)
		attn_output = attn_output.masked_fill(torch.isnan(attn_output), 0)
		y = self.add_norm(attn_output, y)
		ffn_output = self.ffn(y)
		y = self.add_norm(ffn_output, y)
		return y
