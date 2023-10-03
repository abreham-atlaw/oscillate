import typing

import numpy as np
from bpemb import BPEmb

from oscillate.data.prepare.encoders.encoder import Encoder


class BpembEncoder(Encoder):

	def __init__(self, lang: str = "en", emb_size: int = 50):
		self.__model = BPEmb(lang=lang, dim=emb_size)
		self.__emb_size = emb_size

	def encode(self, inputs: typing.Any) -> np.ndarray:
		tokens = np.array(self.__model.embed(inputs))
		return tokens

	def get_embedding_size(self) -> int:
		return self.__emb_size
