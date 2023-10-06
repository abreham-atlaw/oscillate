import unittest

import torch

from oscillate.model.model.decoder import Decoder


class DecoderTest(unittest.TestCase):

	def test_functionality(self):

		BLOCK_SIZE = 5
		EMB_SIZE = 6
		BATCH_SIZE = 4

		X = torch.rand((BATCH_SIZE, BLOCK_SIZE, EMB_SIZE))
		X_enc = torch.rand((BATCH_SIZE, BLOCK_SIZE, EMB_SIZE))
		decoder = Decoder(
			block_size=BLOCK_SIZE,
			emb_size=EMB_SIZE,
			num_heads=3,
			ff_size=1024
		)

		y = decoder(X, X_enc)
		self.assertEqual(X.shape, y.shape)
