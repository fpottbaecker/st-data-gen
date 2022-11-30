from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from tqdm.auto import tqdm
from os import stat
from os.path import basename, isfile
from functools import partial
from tempfile import gettempdir

BLOCKSIZE=2**13


def download(url, /, directory=None, filename=None):
    directory = directory or gettempdir()
    filename = filename or basename(unquote(urlparse(url).path))
    path = directory + "/" + filename
    
    with tqdm(desc=filename, unit="iB", unit_scale=True, unit_divisor=1024) as progress:
        progress.set_postfix(status="connecting")
        with urlopen(url) as r:
            size = int(r.headers.get('content-length', 0))
            progress.reset(total=size)
            progress.set_postfix(status="connected")
            if isfile(path) and size == stat(path).st_size:
                progress.update(size)
                progress.set_postfix(status="exists")
                return path
            progress.set_postfix(status="downloading")
            with open(path, "wb") as f:
                for chunk in iter(partial(r.read, BLOCKSIZE), b""):
                    progress.update(f.write(chunk))
            progress.set_postfix(status="downloaded")
    
    return path
