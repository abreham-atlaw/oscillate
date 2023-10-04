import random
from collections import OrderedDict

import numpy as np
import torch
import typing
from torch.utils.data import Dataset

import os


class TTADataset(Dataset):
	def __init__(
			self,
			root_dir,
			cache_size: int = 5,
			X_encoder_dir: str = "X_encoder",
			X_decoder_dir: str = "X_decoder",
			y_dir: str = "y",
			out_dtype: typing.Type = np.float32
	):
		self.__dtype = out_dtype
		self.root_dir = root_dir
		self.__X_encoder_dir = os.path.join(root_dir, X_encoder_dir)
		self.__X_decoder_dir = os.path.join(root_dir, X_decoder_dir)
		self.__y_dir = os.path.join(root_dir, y_dir)

		self.files = sorted(os.listdir(self.__X_decoder_dir))
		self.cache = OrderedDict()
		self.cache_size = cache_size
		self.data_points_per_file = self.__get_dp_per_file()

	def shuffle(self):
		random.shuffle(self.files)
		self.cache = OrderedDict()

	def __get_dp_per_file(self) -> int:
		first_file_name = self.files[0]
		first_file_data = self.__load_array(os.path.join(self.__X_decoder_dir, first_file_name))
		return first_file_data.shape[0]

	def __len__(self):
		return len(self.files) * self.data_points_per_file

	def __load_array(self, path: str) -> np.ndarray:
		return np.load(path).astype(self.__dtype)

	def __getitem__(self, idx):
		file_idx = idx // self.data_points_per_file
		data_idx = idx % self.data_points_per_file

		if file_idx not in self.cache:
			if len(self.cache) >= self.cache_size:
				self.cache.popitem(last=False)

			file_name = self.files[file_idx]
			X_enc = self.__load_array(os.path.join(self.root_dir, 'X_encoder', file_name))
			X_dec = self.__load_array(os.path.join(self.root_dir, 'X_decoder', file_name))
			y = self.__load_array(os.path.join(self.root_dir, 'y', file_name))

			self.cache[file_idx] = (X_enc, X_dec, y)

		return tuple([torch.from_numpy(x[data_idx]) for x in self.cache[file_idx]])
