import typing
from abc import ABC, abstractmethod

import numpy as np


class Encoder(ABC):

	@abstractmethod
	def encode(self, inputs: typing.Any) -> np.ndarray:
		pass

	@abstractmethod
	def get_embedding_size(self) -> int:
		pass

	def get_eos_token(self) -> int:
		return 0
