import torch.nn as nn

from oscillate.model.model.decoder import Decoder
from oscillate.model.model.encoder import Encoder


class TTAModel(nn.Module):

	def __init__(self, encoder: Encoder, decoder: Decoder, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.encoder = encoder
		self.decoder = decoder
		self.enc_reshape = nn.Linear(self.encoder.emb_size, self.decoder.emb_size)

	def forward(self, X_encoder, X_decoder):
		y_encoder, pad_mask = self.encoder(X_encoder)
		y_encoder = self.enc_reshape(y_encoder)
		y_decoder = self.decoder(X_decoder, y_encoder, pad_mask)
		return y_decoder
