import torch
import torch.nn as nn

from oscillate.model.model.decoder import Decoder
from oscillate.model.model.encoder import Encoder


class TTAModel(nn.Module):

	def __init__(self, encoder: Encoder, decoder: Decoder, *args, decoder_vocab_size=1024, dtype=torch.float32, **kwargs):
		super().__init__(*args, **kwargs)
		self.encoder = encoder
		self.decoder = decoder
		self.enc_reshape = nn.Linear(self.encoder.emb_size, self.decoder.emb_size, dtype=dtype)
		self.dec_reshape = nn.Linear(self.decoder.emb_size, decoder_vocab_size * self.decoder.input_emb_size, dtype=dtype)
		self.softmax = nn.Softmax(-1)
		self.decoder_vocab_size = decoder_vocab_size

	def forward(self, X_encoder, X_decoder):
		y_encoder, pad_mask = self.encoder(X_encoder)
		y_encoder = self.enc_reshape(y_encoder)
		y_decoder = self.decoder(X_decoder, y_encoder, pad_mask)
		y = self.dec_reshape(y_decoder)
		y = self.softmax(y).reshape((*X_decoder.shape, self.decoder_vocab_size))
		return y
