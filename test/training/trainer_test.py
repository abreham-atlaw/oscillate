import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from oscillate.data.load.dataset import TTADataset
from oscillate.data.load.processors.one_hot_processor import OneHotProcessor
from oscillate.model.model.decoder import Decoder
from oscillate.model.model.encoder import Encoder
from oscillate.model.model.model import TTAModel
from oscillate.training.trainer import TTATrainer


class TTATrainerTest(unittest.TestCase):

	def test_functionality(self):
		BLOCK_SIZE = 512
		ENCODER_EMB_SIZE = 50
		DECODER_INPUT_EMB_SIZE = 4
		DECODER_EMB_SIZE = 64
		ENCODER_HEADS = 50
		DECODER_HEADS = 32
		FF_SIZE = 1024
		DECODER_VOCAB_SIZE = 1024
		DTYPE = torch.float32
		NP_DTYPE = np.float32

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

		dataset = TTADataset(
			root_dirs=[
				"/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/dummy/prepared/train",
				# "/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/dummy/prepared1/train",
			],
			out_dtypes=NP_DTYPE,
		)
		dataloader = DataLoader(dataset, batch_size=4)

		loss_function = nn.CrossEntropyLoss()
		optimizer = Adam(model.parameters(), lr=0.001)

		trainer = TTATrainer(model, loss_function=loss_function, optimizer=optimizer)
		trainer.train(dataloader, epochs=3, progress=True)
