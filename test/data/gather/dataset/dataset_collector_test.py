import unittest

import pandas as pd

from oscillate.data.gather.dataset.dataset_collector import DatasetCollector


class DatasetCollectorTest(unittest.TestCase):

	def test_functionality(self):
		collector = DatasetCollector(
			"/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/collected",
			checkpoint=3
		)
		df = pd.read_csv("/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/charts-1k.csv")
		collector.collect_from_df(
			df,
			header_title="song"
		)
