import typing

import numpy as np

from .processor import TTAProcessor


class OneHotProcessor(TTAProcessor):

	def __init__(self, vocab_size: int):
		super().__init__()
		self.__vocab_size = vocab_size

	def __one_hot_encode(self, y: np.ndarray) -> np.ndarray:
		y_sm = np.zeros((y.shape[0], self.__vocab_size), dtype=np.int8)
		y_sm[np.arange(y.shape[0]), y] = 1
		return y_sm

	def __one_hot_encode_sequence(self, y: np.ndarray) -> np.ndarray:
		y_flat = y.flatten()
		encoded = self.__one_hot_encode(y_flat)
		encoded = encoded.reshape((*y.shape, self.__vocab_size))
		return encoded

	def process(self, X_enc: np.ndarray, X_dec: np.ndarray, y: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
		return X_enc, X_dec, self.__one_hot_encode_sequence(y.astype(np.int8)).astype(y.dtype)
