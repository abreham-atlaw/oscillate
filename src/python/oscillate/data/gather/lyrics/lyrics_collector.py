from abc import ABC, abstractmethod


class LyricsCollector(ABC):

	@abstractmethod
	def collect_lyrics(self, title: str, artist: str) -> str:
		pass
