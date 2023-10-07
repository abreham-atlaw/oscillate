import unittest

import numpy as np

from oscillate.data.load.processors.one_hot_processor import OneHotProcessor


class OneHotProcessorTest(unittest.TestCase):

	def test_functionality(self):
		processor = OneHotProcessor(30)
		X_enc, X_dec, y = np.random.random((10, 5)), np.random.random((10, 3)), np.arange(30).reshape((10, 3))
		_, __, n_y = processor.process(X_enc, X_dec, y)
		self.assertEqual(n_y.shape, (*y.shape, 30))
