import typing

import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

from oscillate.data.prepare.encoders.encoder import Encoder


class EncodecEncoder(Encoder):

	def __init__(self, bandwidth: float= 12.0, eos_token: int = 0):
		self.__model = EncodecModel.encodec_model_24khz()
		self.__model.set_target_bandwidth(bandwidth)
		self.__eos_token = eos_token

	def encode(self, inputs: typing.Any) -> np.ndarray:
		wav, sr = torchaudio.load(inputs)
		wav = convert_audio(wav, sr, self.__model.sample_rate, self.__model.channels)
		wav = wav.unsqueeze(0)

		with torch.no_grad():
			tokens = self.__model.encode(wav)

		tokens = tokens[0][0].numpy().squeeze()
		eos_token = np.zeros((tokens.shape[0], 1)) + self.__eos_token
		tokens = np.transpose(np.concatenate((tokens, eos_token), axis=1)) + 1024
		return tokens.astype(np.int16)

	def get_embedding_size(self) -> int:
		return int(self.__model.bandwidth * 4/3)
