import unittest

import torch

from oscillate.model.model.decoder import Decoder


class DecoderTest(unittest.TestCase):

	def test_functionality(self):

		BLOCK_SIZE = 5
		EMB_SIZE = 6
		INPUT_EMB_SIZE = 3
		BATCH_SIZE = 4

		X = torch.randint(1, 1024, (BATCH_SIZE, BLOCK_SIZE, INPUT_EMB_SIZE))
		X_enc = torch.rand((BATCH_SIZE, BLOCK_SIZE, EMB_SIZE))
		enc_mask = torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.bool)
		decoder = Decoder(
			block_size=BLOCK_SIZE,
			input_emb_size=INPUT_EMB_SIZE,
			emb_size=EMB_SIZE,
			num_heads=3,
			ff_size=1024
		)

		y = decoder(X, X_enc, enc_mask)
		self.assertEqual(X.shape[:2], y.shape[:2])
		self.assertEqual(y.shape[-1], EMB_SIZE)
