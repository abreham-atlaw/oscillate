import unittest

import numpy as np
import torch
import torch.nn.functional as F

from oscillate.model.model.encoder import Encoder


class EncoderTest(unittest.TestCase):

	def test_functionality(self):
		EMB_SIZE = 6
		BLOCK_SIZE = 5
		BATCH_SIZE = 4
		X = F.pad(torch.rand((BATCH_SIZE, BLOCK_SIZE-2, EMB_SIZE)), (0, 0, 0, 2), "constant", value=0)

		encoder = Encoder(
			input_emb_size=EMB_SIZE,
			block_size=BLOCK_SIZE,
			mha_heads=3,
			ff_size=1024
		)
		y = encoder(X)
		self.assertEqual(y.shape, X.shape)
