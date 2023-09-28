from abc import ABC, abstractmethod

from .metadata import MetaData


class MetadataCollector(ABC):

	@abstractmethod
	def collect_metadata(self, title: str, artist: str) -> MetaData:
		pass
