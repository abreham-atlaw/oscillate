from abc import abstractmethod, ABC

import os


class AudioCollector(ABC):

	def __init__(self, output_path: str = ".", file_extension: str = "m4a"):
		self.__output_path = output_path
		self._file_extension = file_extension

	def set_outpath(self, output_path: str):
		self.__output_path = output_path

	def _generate_filename(self, title: str, artist: str) -> str:
		return f"{title.replace(' ', '_')}-{artist.replace(' ', '_')}.{self._file_extension}"

	@abstractmethod
	def _collect_audio(self, title: str, artist: str, output_path: str):
		pass

	def collect_audio(self, title: str, artist: str) -> str:
		filename = self._generate_filename(title, artist)
		self._collect_audio(title, artist, os.path.join(self.__output_path, filename))
