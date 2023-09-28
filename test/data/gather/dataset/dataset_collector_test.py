import unittest

import pandas as pd

from oscillate.data.gather.dataset.dataset_collector import DatasetCollector


class DatasetCollectorTest(unittest.TestCase):

	def test_functionality(self):
		collector = DatasetCollector(
			"/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/collected",
		)
		df = pd.read_csv("/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/charts-20.csv")
		collector.collect_from_df(
			df,
			header_title="song"
		)
