import os.path

from pytube import YouTube, Search

from oscillate.data.gather.audio.audio_collector import AudioCollector

import yt_dlp as youtube_dl


class YoutubeCollector(AudioCollector):

	@staticmethod
	def __search(title: str, artist: str) -> str:
		search = Search(f"{artist} {title} Audio")
		top_result: YouTube = search.results[0]
		return top_result.watch_url

	def __download(self, url, output_path: str):
		filename, path = os.path.basename(output_path), os.path.dirname(output_path)
		ydl_opts = {
			"format": self._file_extension,
			"paths": {
				"home": path},
			"outtmpl": filename
		}
		with youtube_dl.YoutubeDL(ydl_opts) as ydl:
			ydl.download([url])

	def _collect_audio(self, title: str, artist: str, output_path: str):
		url = self.__search(title, artist)
		self.__download(url, output_path)
