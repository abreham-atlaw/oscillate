import unittest

from oscillate.data.prepare.encoders.text.bpemb_encoder import BpembEncoder


class BpembEncoderTest(unittest.TestCase):

	def test_functionality(self):
		EMB_SIZE = 50
		TEXT = "The quick brown fox jumps over the lazy dog."

		encoder = BpembEncoder(
			lang="en",
			emb_size=EMB_SIZE
		)

		tokens = encoder.encode(TEXT)

		self.assertEqual(
			tokens.shape[1],
			EMB_SIZE
		)
