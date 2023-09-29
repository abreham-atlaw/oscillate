import os.path
import typing

import pandas as pd

from oscillate.di.data import DataProviders
from .track import Track
from oscillate.data.gather.metadata import MetaData


class DatasetCollector:

	def __init__(
			self,
			out_path: str,
			audio_dir="audio",
			csv_filename: str = "data.csv",
			header_lyrics: str = "lyrics",
			header_audio: str = "audio",
			header_genre: str = "genre",
			header_title: str = "title",
			header_artist: str = "artist",
			skip_lyrics: bool = False,
			skip_audio: bool = False,
			skip_metadata: bool = False,
			checkpoint: int = 1000
	):
		self.__out_path = out_path
		self.__audio_path = os.path.join(self.__out_path, audio_dir)
		self.__csv_path = os.path.join(self.__out_path, csv_filename)
		if not os.path.exists(self.__audio_path):
			os.mkdir(self.__audio_path)
		self.__header_lyrics = header_lyrics
		self.__header_audio = header_audio
		self.__header_genre = header_genre
		self.__header_title = header_title
		self.__header_artist = header_artist
		self.__columns = [
			self.__header_title,
			self.__header_artist,
			self.__header_genre,
			self.__header_audio,
			self.__header_lyrics
		]

		self.__audio_collector = DataProviders.provide_audio_collector()
		self.__lyrics_collector = DataProviders.provide_lyrics_collector()
		self.__metadata_collector = DataProviders.provide_metadata_collector()
		self.__audio_collector.set_outpath(self.__audio_path)
		self.__skip_lyrics, self.__skip_audio, self.__skip_metadata = skip_lyrics, skip_audio, skip_metadata
		self.__checkpoint = checkpoint

	def __save(self, df: pd.DataFrame):
		print("[+]Saving...")
		df.to_csv(self.__csv_path)

	def collect_datapoint(self, track: Track) -> typing.Tuple[str, str, MetaData]:
		if self.__skip_audio:
			audio = ""
		else:
			audio = self.__audio_collector.collect_audio(track.title, track.artist)

		if self.__skip_lyrics:
			lyrics = ""
		else:
			lyrics = self.__lyrics_collector.collect_lyrics(track.title, track.artist)

		if self.__skip_metadata:
			metadata = MetaData(genre="")
		else:
			metadata = self.__metadata_collector.collect_metadata(track.title, track.artist)
		return lyrics, audio, metadata

	def collect(self, tracks: typing.List[Track]):
		df = pd.DataFrame(columns=[
			self.__columns
		])
		for i, track in enumerate(tracks):
			try:
				lyrics, audio, metadata = self.collect_datapoint(track)
			except Exception as ex:
				print(f"[-]Failed to fetch {track.title} by {track.artist} due to : {ex}")
				continue
			df.loc[len(df.index)] = [
				track.title,
				track.artist,
				metadata.genre,
				audio,
				lyrics
			]
			print(f"[+]Collected: {100*(i+1)/len(tracks): .2f}%...")
			if (i - 1) % self.__checkpoint == 0 and i != 1:
				self.__save(df)

		self.__save(df)


	def collect_from_df(self, df: pd.DataFrame, header_title: str = "title", header_artist: str = "artist"):
		tracks = []
		for i, row in df.iterrows():
			tracks.append(Track(
				title=row[header_title],
				artist=row[header_artist]
			))

		return self.collect(tracks)
