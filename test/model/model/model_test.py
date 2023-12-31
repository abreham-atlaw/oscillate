import unittest

import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from torch.utils.data import DataLoader

from oscillate.data.load.dataset import TTADataset
from oscillate.data.prepare.encoders.text.bpemb_encoder import BpembEncoder
from oscillate.model.model.decoder import Decoder
from oscillate.model.model.encoder import Encoder
from oscillate.model.model.model import TTAModel


class TTAModelTest(unittest.TestCase):

	def test_functionality(self):
		BLOCK_SIZE = 512
		ENCODER_EMB_SIZE = 6
		DECODER_EMB_SIZE = 16
		DECODER_INPUT_EMB_SIZE = 8
		MHA_HEADS = 2
		BATCH_SIZE = 16
		DECODER_VOCAB_SIZE = 1024

		model = TTAModel(
			encoder=Encoder(
				block_size=BLOCK_SIZE,
				emb_size=ENCODER_EMB_SIZE,
				ff_size=256,
				mha_heads=MHA_HEADS
			),
			decoder=Decoder(
				emb_size=DECODER_EMB_SIZE,
				input_emb_size=DECODER_INPUT_EMB_SIZE,
				block_size=BLOCK_SIZE,
				num_heads=MHA_HEADS,
				ff_size=256
			),
			decoder_vocab_size=DECODER_VOCAB_SIZE
		)

		X_encoder = torch.rand((BATCH_SIZE, BLOCK_SIZE, ENCODER_EMB_SIZE))
		X_decoder = torch.randint(0, 1024, (BATCH_SIZE, BLOCK_SIZE, DECODER_INPUT_EMB_SIZE))

		y = model(X_encoder, X_decoder)

		self.assertEqual(y.shape, (X_decoder.shape[0], DECODER_INPUT_EMB_SIZE, DECODER_VOCAB_SIZE))

	def test_load_and_run(self):
		def get_X_enc(text):
			X = text_encoder.encode("Hello. How are you?")
			X = np.pad(X, pad_width=((0, 512 - X.shape[0]), (0, 0)), mode="constant", constant_values=0)
			return torch.from_numpy(X).unsqueeze(0)

		ENCODEC_BANDWIDTH = 3
		ENCODEC_EOS_TOKEN = 0
		BPEMB_LANG = "en"
		BPEMB_EMB_SIZE = 50
		BLOCK_SIZE = 512
		ENCODER_EMB_SIZE = 50
		DECODER_INPUT_EMB_SIZE = 4
		DECODER_EMB_SIZE = 256
		ENCODER_HEADS = 50
		DECODER_HEADS = 128
		FF_SIZE = 1024
		DECODER_VOCAB_SIZE = 1024

		DTYPE = torch.float32
		NP_DTYPE = np.float32

		text_encoder = BpembEncoder(lang=BPEMB_LANG, emb_size=BPEMB_EMB_SIZE)
		encodec_model = EncodecModel.encodec_model_24khz()
		encodec_model.set_target_bandwidth(ENCODEC_BANDWIDTH)

		X_enc = get_X_enc("The quick brown fox jumps over the lazy dog")
		X_dec = torch.zeros((1, 512, DECODER_INPUT_EMB_SIZE))

		model = TTAModel(
			encoder=Encoder(
				block_size=BLOCK_SIZE,
				emb_size=ENCODER_EMB_SIZE,
				ff_size=FF_SIZE,
				mha_heads=ENCODER_HEADS,
				dtype=DTYPE
			),
			decoder=Decoder(
				emb_size=DECODER_EMB_SIZE,
				input_emb_size=DECODER_INPUT_EMB_SIZE,
				block_size=BLOCK_SIZE,
				num_heads=DECODER_HEADS,
				ff_size=FF_SIZE,
				dtype=DTYPE,
				vocab_size=DECODER_VOCAB_SIZE
			),
			dtype=DTYPE,
			decoder_vocab_size=DECODER_VOCAB_SIZE
		)
		model.load_state_dict(torch.load('/home/abreham/Projects/TeamProjects/Oscillate/temp/models/model.pth', map_location=torch.device('cpu')))
		model.eval()
		out = torch.zeros_like(X_dec)[0]

		with torch.no_grad():
			for i in range(64):
				y = model(X_enc, X_dec)
				out[i] = torch.argmax(y, dim=-1)[0, -1]
				X_dec[0, 512-(i+1):] = out[:i+1]
				# out = torch.argmax(y, dim=-1)[0]
				# X_dec[0] = out

		X_np = out.detach().numpy()
		X_np = X_np[np.any(X_np != 0, axis=1)]

		audio_codes = [
			(
				torch.from_numpy(
					np.round(
						np.transpose(
							X_np
						)
					).astype(int),
				).unsqueeze(0),
				None
			)
		]
		with torch.no_grad():
			audio_values = encodec_model.decode(audio_codes)[0]
		torchaudio.save("/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/test.wav", audio_values, encodec_model.sample_rate)
