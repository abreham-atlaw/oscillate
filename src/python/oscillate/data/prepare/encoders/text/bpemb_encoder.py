import typing

import numpy as np
from bpemb import BPEmb

from oscillate.data.prepare.encoders.encoder import Encoder


class BpembEncoder(Encoder):

	def __init__(self, lang: str = "en", emb_size: int = 50, logging=False):
		self.__model = BPEmb(lang=lang, dim=emb_size)
		self.__emb_size = emb_size
		self.__logging = logging

	def __log(self, text):
		if not self.__logging:
			return
		print(text)

	def encode(self, inputs: typing.Any) -> np.ndarray:
		self.__log(f"Encoding \"{inputs}\"")
		tokens = np.array(self.__model.embed(inputs))
		return tokens

	def get_embedding_size(self) -> int:
		return self.__emb_size
