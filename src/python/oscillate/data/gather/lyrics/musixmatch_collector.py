from musixmatch import Musixmatch

from oscillate.data.gather.lyrics.lyrics_collector import LyricsCollector


class MusixMatchCollector(LyricsCollector):

	def __init__(self, apikey, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__musixmatch = Musixmatch(apikey=apikey)

	def collect_lyrics(self, title: str, artist: str) -> str:
		lyrics = self.__musixmatch.matcher_lyrics_get(q_track="Easy On Me", q_artist="Adele")
		return lyrics["message"]["body"]["lyrics"]["lyrics_body"]
