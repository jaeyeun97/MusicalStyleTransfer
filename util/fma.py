import os
import pandas as pd
import ast

"""
Dealing with the metadata of the dataset.
Based on the code from the FMA Dataset Github
"""

class FMA(object):
    def __init__(self, metadata_dir, audio_dir):
        self.matadata_dir = metadata_dir
        self.audio_dir = audio_dir

        if audio_path.endswith('fma_small'):
            self.audio_type = 'small'
        elif audio_path.endswith('fma_medium'):
            self.audio_type = 'medium'
        elif audio_path.endswith('fma_large'):
            self.audio_type = 'large'
        elif audio_path.endswith('fma_full'):
            self.audio_type = 'full'
        else:
            raise Exception('Data directory name should start with "fma_".')

        self.genres = self.load_genre()
        self.tracks = self.load_track()
        if self.audio_type != 'full':
            self.tracks = self.tracks[self.tracks['set', 'subset'] <= self.audio_type]

    def get_all_genres(self):
        return self.genres['title'].tolist()

    def get_genre_id(self, genre_name):
        return self.genres.index[self.genres['title'] == genre_name].item()

    def get_track_ids_by_genre(self, genre_id):
        """
        @param genre_id (pd.Index): Genre ID
        @returns tracks (pd.Series): Tracks of said genre.
        """
        mask = self.tracks['track', 'genres_all'].apply(lambda row: genre_id in row)
        return self.tracks[mask].index

    def get_audio_path(self, track_id):
        tid_str = '{:06d}'.format(track_id)
        return os.path.join(self.audio_dir, tid_str[:3], tid_str + '.mp3')

    def load_genre(self):
        return pd.read_csv(os.path.join(self.matadata_dir, 'genres.csv'), index_col=0)

    def load_track(filepath):
        tracks = pd.read_csv(os.path.join(self.matadata_dir, 'tracks.csv'), index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                'category', categories=SUBSETS, ordered=True)

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks
