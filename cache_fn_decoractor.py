import os
import pickle
from pathlib import Path

CACHE_DIR = Path(__file__).parent / 'cache'


def cache(cache_id):
    path = CACHE_DIR / cache_id
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir()

    def decorator(fn):
        def wrapped(*args, **kwargs):
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    print(f'Restoring cache : {path}')
                    return pickle.load(f)
            res = fn(*args, **kwargs)
            with open(path, 'wb') as f:
                print(f'Saving cache : {path}')
                pickle.dump(res, f)
            return res

        return wrapped

    return decorator
