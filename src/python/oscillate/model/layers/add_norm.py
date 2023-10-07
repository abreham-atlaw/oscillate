import torch
import torch.nn as nn


class AddNorm(nn.Module):

	def __init__(self, shape, *args, dtype=torch.float32, **kwargs):
		super().__init__(*args, **kwargs)
		self.norm = nn.LayerNorm(shape, dtype=dtype)

	def forward(self, X, residual):
		sum = X + residual
		return self.norm(sum)
