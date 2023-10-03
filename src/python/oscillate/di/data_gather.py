from oscillate.data.gather.audio import AudioCollector, YoutubeCollector
from oscillate.data.gather.lyrics import LyricsCollector, GeniusCollector

from oscillate.configs import GENIUS_APIKEY, MUSIXMATCH_APIKEY
from oscillate.data.gather.metadata import MetadataCollector, MusixmatchMetadataCollector


class DataGatheringProviders:

	@staticmethod
	def provide_lyrics_collector() -> LyricsCollector:
		return GeniusCollector(GENIUS_APIKEY)

	@staticmethod
	def provide_audio_collector() -> AudioCollector:
		return YoutubeCollector()

	@staticmethod
	def provide_metadata_collector() -> MetadataCollector:
		return MusixmatchMetadataCollector(MUSIXMATCH_APIKEY)
