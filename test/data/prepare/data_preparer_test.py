import unittest

import numpy as np
import torch
import pandas as pd
import torchaudio
from encodec import EncodecModel

from oscillate.data.prepare.data_preparer import DataPreparer
from oscillate.data.prepare.encoders.audio import EncodecEncoder
from oscillate.data.prepare.encoders.text.bpemb_encoder import BpembEncoder
from oscillate.model.model.decoder import Decoder
from oscillate.model.model.encoder import Encoder
from oscillate.model.model.model import TTAModel


class DataPreparerTest(unittest.TestCase):

	def test_functionality(self):
		audio_encoder = EncodecEncoder()
		text_encoder = BpembEncoder()

		preparer = DataPreparer(
			audio_encoder,
			text_encoder,
			block_size=512,
			audio_dir="/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/dummy/audio",
			audio_file_format="{}.wav",
			checkpoint=500
		)

		df = pd.read_csv("/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/dummy/metadata.csv", sep="|")

		preparer.start(
			df=df,
			save_path="/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/dummy/prepared1",
			header_text="text",
			header_audio="audio",
			export_remaining=False
		)
