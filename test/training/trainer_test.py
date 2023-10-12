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
		DECODER_EMB_SIZE = 128
		DECODER_INPUT_EMB_SIZE = 16
		DECODER_VOCAB_SIZE = 1024
		MHA_HEADS = 2
		DTYPE = torch.float32
		NP_DTYPE = np.float32

		model = TTAModel(
			encoder=Encoder(
				block_size=BLOCK_SIZE,
				emb_size=ENCODER_EMB_SIZE,
				ff_size=256,
				mha_heads=MHA_HEADS,
				dtype=DTYPE,
			),
			decoder=Decoder(
				emb_size=DECODER_EMB_SIZE,
				input_emb_size=DECODER_INPUT_EMB_SIZE,
				block_size=BLOCK_SIZE,
				num_heads=MHA_HEADS,
				ff_size=256,
				vocab_size=DECODER_VOCAB_SIZE,
				dtype=DTYPE
			),
			dtype=DTYPE
		)

		dataset = TTADataset(
			root_dirs=[
				"/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/dummy/prepared/train",
				"/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/dummy/prepared1/train",
			],
			out_dtypes=NP_DTYPE,
			processors=[
				OneHotProcessor(DECODER_VOCAB_SIZE)
			]
		)
		dataloader = DataLoader(dataset, batch_size=4)

		loss_function = nn.CrossEntropyLoss()
		optimizer = Adam(model.parameters(), lr=0.001)

		trainer = TTATrainer(model, loss_function=loss_function, optimizer=optimizer)
		trainer.train(dataloader, epochs=3, progress=True)
