import typing
from abc import ABC, abstractmethod

import numpy as np


class TTAProcessor(ABC):

	@abstractmethod
	def process(self, X_enc: np.ndarray, X_dec: np.ndarray, y: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
		pass
