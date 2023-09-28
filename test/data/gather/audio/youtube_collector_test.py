import unittest

from oscillate.data.gather.audio import YoutubeCollector


class YoutubeCollectorTest(unittest.TestCase):

	def test_functionality(self):
		collector = YoutubeCollector(output_path="/home/abreham/Projects/TeamProjects/Oscillate/temp/Data/collected/audio")
		collector.collect_audio(

		)
