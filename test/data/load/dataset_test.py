import unittest

import numpy as np
import torch

from oscillate.data.load.dataset import TTADataset
from oscillate.data.load.processors.one_hot_processor import OneHotProcessor


class DatasetTest(unittest.TestCase):

	def test_functionality(self):

		processors = [
			OneHotProcessor(1024)
		]
		dataset = TTADataset(
			[
				"/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/dummy/prepared/train",
				"/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/dummy/prepared1/train"
			],
			processors=processors
		)
		X_enc, X_dec, y = dataset[3400]
		self.assertIsInstance(X_enc, torch.Tensor)
		self.assertEqual(X_dec.shape[0], X_enc.shape[0])
		self.assertEqual(X_enc.shape[0], y.shape[0])
