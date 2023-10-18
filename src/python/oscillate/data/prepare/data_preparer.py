import typing

import os
from datetime import datetime
import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from oscillate.data.prepare.encoders.encoder import Encoder


class DataPreparer:

	def __init__(
			self,

			audio_encoder: Encoder,
			text_encoder: Encoder,

			audio_dir: str,

			audio_file_format: str = "{}",

			audio_padding_token: int = 0,
			text_padding_token: int = 0,
			block_size: int = 2049,
			audio_vocab_size: int = 1024,


			train_dir_name: str = "train",
			test_dir_name: str = "test",
			X_encoder_dir_name: str = "X_encoder",
			X_decoder_dir_name: str = "X_decoder",
			y_dir_name: str = "y",

			split_test_size: float = 0.2,
			checkpoint: typing.Optional[int] = None

	):
		self.__audio_padding_token = audio_padding_token
		self.__text_padding_value = text_padding_token

		self.__block_size = block_size

		self.__audio_encoder = audio_encoder
		self.__text_encoder = text_encoder

		self.__audio_dir = audio_dir
		self.__audio_file_format = audio_file_format

		self.__train_dir_name, self.__test_dir_name = train_dir_name, test_dir_name
		self.__X_encoder_dir_name, self.__X_decoder_dir_name, self.__y_dir_name = X_encoder_dir_name, X_decoder_dir_name, \
			y_dir_name

		self.__split_test_size = split_test_size
		self.__batch_size = checkpoint
		self.__audio_vocab_size = audio_vocab_size

	@staticmethod
	def __pad(arr: np.ndarray, block_size: int, pad_token: int) -> np.ndarray:
		pad_size = block_size - arr.shape[0]
		pad_width = ((0, pad_size), (0, 0))
		return np.pad(arr, pad_width=pad_width, mode="constant", constant_values=pad_token)

	@staticmethod
	def __shift_array(arr: np.ndarray, shift: int, block_size: int, pad_token: int) -> np.ndarray:
		new_arr = np.zeros((block_size, arr.shape[1]), dtype=np.int16) + pad_token
		new_arr[-min(shift, block_size):] = arr[max(shift-block_size, 0):shift]
		return new_arr

	@staticmethod
	def __generate_filename() -> str:
		return f"{datetime.now().timestamp()}.npy"

	def __one_hot_encode(self, y: np.ndarray) -> np.ndarray:
		y_sm = np.zeros((y.shape[0], self.__audio_vocab_size), dtype=np.int16)
		y_sm[np.arange(y.shape[0]), y] = 1
		return y_sm

	def __one_hot_encode_sequence(self, y: np.ndarray) -> np.ndarray:
		y_flat = y.flatten()
		encoded = self.__one_hot_encode(y_flat)
		encoded = encoded.reshape((*y.shape, self.__audio_vocab_size))
		return encoded

	def __encode_audio(self, audio: str) -> np.ndarray:
		filepath = os.path.join(self.__audio_dir, self.__audio_file_format.format(audio))
		return self.__audio_encoder.encode(filepath)

	def __encode_text(self, text: str) -> np.ndarray:
		return self.__text_encoder.encode(text)

	def __prepare_row(self, audio: str, text: str) -> typing.Tuple[typing.Tuple[np.ndarray, np.ndarray], np.ndarray]:
		encoded_text = self.__encode_text(text)
		encoded_text = self.__pad(
			encoded_text,
			self.__block_size,
			self.__text_padding_value
		)

		encoded_audio = self.__encode_audio(audio).astype(np.int16)
		shifted_audios = np.stack([
			self.__shift_array(
				encoded_audio,
				i+1,
				self.__block_size,
				self.__audio_padding_token
			)
			for i in range(encoded_audio.shape[0])
		], axis=0)

		X_decoder = shifted_audios[:-1]
		X_encoder = np.repeat(
			np.expand_dims(
				encoded_text,
				axis=0
			),
			X_decoder.shape[0],
			axis=0
		)

		y = shifted_audios[1:, -1]
		y = self.__one_hot_encode_sequence(y)
		return (X_encoder, X_decoder), y

	def __setup_save_path(
			self, path: str):
		if not os.path.exists(path):
			os.mkdir(path)

		role_paths = [
			os.path.join(path, role_dir_name)
			for role_dir_name in [self.__train_dir_name, self.__test_dir_name]
		]
		storage_paths = [
			os.path.join(role_path, storage_dir_name)
			for role_path in role_paths
			for storage_dir_name in [self.__X_encoder_dir_name, self.__X_decoder_dir_name, self.__y_dir_name]
		]

		for storage_path in storage_paths:
			if not os.path.exists(storage_path):
				os.makedirs(storage_path)

	@staticmethod
	def __save_array(arr: np.ndarray, path: str):
		np.save(path, arr)

	def __save(
			self,
			X_encoder: np.ndarray,
			X_decoder: np.ndarray,
			y: np.ndarray,
			path: str,
			is_test: bool
	) -> str:
		filename = self.__generate_filename()
		role_dir_name = self.__train_dir_name
		if is_test:
			role_dir_name = self.__test_dir_name

		role_path = os.path.join(path, role_dir_name)

		for arr, dir_name in zip(
				[X_encoder, X_decoder, y],
				[self.__X_encoder_dir_name, self.__X_decoder_dir_name, self.__y_dir_name]
		):
			save_path = os.path.join(role_path, dir_name, filename)
			self.__save_array(arr, save_path)

		return filename

	def __split_and_save(self, X_encoder: np.ndarray, X_decoder: np.ndarray, y: np.ndarray, path: str):
		data_len = X_encoder.shape[0]

		indices = train_test_split(np.arange(data_len), test_size=self.__split_test_size)

		for role_indices, is_test in zip(indices, [False, True]):
			self.__save(
				X_encoder=X_encoder[role_indices],
				X_decoder=X_decoder[role_indices],
				y=y[role_indices],
				path=path,
				is_test=is_test
			)

	def __checkpoint(self, X_encoder: np.ndarray, X_decoder: np.ndarray, y: np.ndarray, save_path: str) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
		size = X_encoder.shape[0]
		if self.__batch_size is None or size < self.__batch_size:
			return X_encoder, X_decoder, y

		to_save = [arr[:self.__batch_size] for arr in [X_encoder, X_decoder, y]]
		self.__split_and_save(*to_save, path=save_path)
		del to_save
		gc.collect()

		remaining = [arr[self.__batch_size:] for arr in [X_encoder, X_decoder, y]]
		return self.__checkpoint(*remaining, save_path=save_path)

	def start(
			self,
			df: pd.DataFrame,
			save_path: str,
			header_audio: str = "audio",
			header_text: str = "text",
			export_remaining: bool = True
	):
		if not export_remaining and self.__batch_size is None:
			print("Warning: export_remaining set to false without batch_size. No exports will be made.")

		df = df.dropna()

		self.__setup_save_path(save_path)

		X_encoder, X_decoder, y = None, None, None
		for i, row in df.iterrows():

			audio, text = row[header_audio], row[header_text]
			(X_encoder_row, X_decorder_row), y_row = self.__prepare_row(audio, text)

			if X_encoder is None:
				X_encoder, X_decoder, y = X_encoder_row, X_decorder_row, y_row
			else:
				X_encoder, X_decoder, y = [
					np.concatenate([old, new], axis=0)
					for (old, new) in zip([X_encoder, X_decoder, y], [X_encoder_row, X_decorder_row, y_row])
				]
			del X_encoder_row, X_decorder_row, y_row
			gc.collect()

			X_encoder, X_decoder, y = self.__checkpoint(X_encoder, X_decoder, y, save_path)
			gc.collect()
			print(f"[+]Preparing: {(i+1)*100/df.shape[0] :.2f}% ...", end="\r")

		if export_remaining:
			self.__split_and_save(
				X_encoder,
				X_decoder,
				y,
				save_path
			)
