from musixmatch import Musixmatch

from oscillate.data.gather.metadata.metadata import MetaData
from oscillate.data.gather.metadata.metadata_collector import MetadataCollector


class MusixmatchMetadataCollector(MetadataCollector):

	def __init__(self, apikey, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__musixmatch = Musixmatch(apikey=apikey)

	def collect_metadata(self, title: str, artist: str) -> MetaData:
		track = self.__musixmatch.matcher_track_get(q_track="Easy On Me", q_artist="Adele")
		return MetaData(
			genre=track["message"]["body"]["track"]["primary_genres"]["music_genre_list"][0]["music_genre"]["music_genre_name"]
		)
