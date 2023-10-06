import unittest

from oscillate.data.prepare.encoders.audio import EncodecEncoder


class EncodecEncoderTest(unittest.TestCase):

	def test_functionality(self):
		BANDWIDTH = 12.0
		TEST_AUDIO_PATH = "/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/collected/audio/Easy_On_Me-Adele.wav"

		encoder = EncodecEncoder(
			bandwidth=BANDWIDTH,
			eos_token=0
		)

		tokens = encoder.encode(TEST_AUDIO_PATH)
		self.assertEqual(
			tokens.shape[1],
			encoder.get_embedding_size()
		)



