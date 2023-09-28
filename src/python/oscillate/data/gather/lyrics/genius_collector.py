import lyricsgenius

from oscillate.data.gather.lyrics.lyrics_collector import LyricsCollector


class GeniusCollector(LyricsCollector):

	def __init__(self, apikey: str, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__genius = lyricsgenius.Genius(apikey)

	def collect_lyrics(self, title: str, artist: str) -> str:
		song = self.__genius.search_song(title, artist)
		return song.lyrics
