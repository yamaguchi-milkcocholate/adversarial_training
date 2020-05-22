from __future__ import annotations
from typing import List
from src.storage.datasetrepo import BaseRepository
import os
import shutil


class ResultsRepository(BaseRepository):
    DIR_NAME = 'results'

    @classmethod
    def make_results_folder(cls, folder_name: str):
        path = os.path.join(os.path.dirname(__file__), cls.DIR_NAME + '/{0}'.format(folder_name))
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)

    @classmethod
    def write_items(cls, filename: str, items: List[tuple]):
        path = os.path.join(os.path.dirname(__file__), cls.DIR_NAME + '/{0}.txt'.format(filename))
        items = [cls._tuple_to_string(item) for item in items]
        with open(path, mode='w') as f:
            f.writelines(items)

    @classmethod
    def _tuple_to_string(cls, item: tuple) -> str:
        return '{0}: {1}'.format(str(item[0]), str(item[1]))
