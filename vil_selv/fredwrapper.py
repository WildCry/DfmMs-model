import os
import pandas as pd
from fredapi import Fred
import pickle
import datetime


class FetchFred:
    def __init__(self, api_key=None, cache_dir='fred_cache') -> None:
        self.key = Fred(
            api_key if api_key is not None else os.getenv('FRED_API_KEY'))
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_path(self, series_id):
        return os.path.join(self.cache_dir, f"{series_id}.pkl")

    def _is_cache_valid(self, cache_path, max_age_days=1):
        if os.path.exists(cache_path):
            mod_time = os.path.getmtime(cache_path)
            current_time = datetime.datetime.now().timestamp()
            return (current_time - mod_time) / 3600 / 24 <= max_age_days

        return False

    def list_cached_series(self):
        """
        Lists all series IDs that are currently cached.
        """
        cached_files = os.listdir(self.cache_dir)
        series_ids = [os.path.splitext(file)[0] for file in cached_files]

        return series_ids

    def check_series_cache_status(self, series_ids, max_age_days=1):
        """
        Checks and reports the cache status of a given series or a list of series.
        :param series_ids: A list of series IDs to check.
        :param max_age_days: Maximum age of the cache in days before it's considered expired.
        :return: A dictionary with series IDs as keys and cache status ('Cached', 'Valid', 'Expired', 'Not Cached') as values.
        """
        status = {}
        for series_id in series_ids:
            cache_path = self._get_cache_path(series_id)
            if os.path.exists(cache_path):
                if self._is_cache_valid(cache_path, max_age_days):
                    status[series_id] = 'Valid'
                else:
                    status[series_id] = 'Expired'
            else:
                status[series_id] = 'Not Cached'

        return status

    def fetch(self, series=None, max_age_days=1) -> pd.DataFrame:
        if series is None:
            series = {'MTS': 'CMRMTSPL', 'PILTP': 'W875RX1',
                      'ENAP': 'PAYEMS', 'IPMAN': 'IPMAN'}

        lst = []
        for series_name, series_id in series.items():
            cache_path = self._get_cache_path(series_id)
            if self._is_cache_valid(cache_path, max_age_days):
                with open(cache_path, 'rb') as f:
                    series_data = pickle.load(f)
            else:
                try:
                    series_data = self.key.get_series(series_id)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(series_data, f)
                except Exception as e:
                    print(f"Failed to fetch data for {series_name}: {e}")
                    continue
            lst.append(series_data.rename(series_name))

        return pd.concat(lst, axis=1)
