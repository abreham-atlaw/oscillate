import unittest

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from oscillate.data.load.dataset import TTADataset
from oscillate.model.model.decoder import Decoder
from oscillate.model.model.encoder import Encoder
from oscillate.model.model.model import TTAModel
from oscillate.training.trainer import TTATrainer


class TTATrainerTest(unittest.TestCase):

	def test_functionality(self):
		BLOCK_SIZE = 512
		ENCODER_EMB_SIZE = 50
		DECODER_EMB_SIZE = 16
		MHA_HEADS = 2

		model = TTAModel(
			encoder=Encoder(
				block_size=BLOCK_SIZE,
				emb_size=ENCODER_EMB_SIZE,
				ff_size=256,
				mha_heads=MHA_HEADS
			),
			decoder=Decoder(
				emb_size=DECODER_EMB_SIZE,
				block_size=BLOCK_SIZE,
				num_heads=MHA_HEADS,
				ff_size=256
			)
		)

		dataset = TTADataset(
			root_dir="/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/dummy/prepared/train",
			out_dtype=np.float32
		)
		dataloader = DataLoader(dataset, batch_size=8)

		loss_function = nn.CrossEntropyLoss()
		optimizer = Adam(model.parameters(), lr=0.001)

		trainer = TTATrainer(model, loss_function=loss_function, optimizer=optimizer)
		trainer.train(dataloader, epochs=3, progress=True)