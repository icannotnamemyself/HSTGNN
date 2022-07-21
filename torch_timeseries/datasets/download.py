import os.path as osp
import sys
from typing import Optional
from tqdm.auto import tqdm

from .makedirs import makedirs
import shutil
import requests

def download_url(url: str, folder: str, log: bool = True,
                 filename: Optional[str] = None):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]
        # filename = os.path.basename(url)

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    makedirs(folder)


    with requests.get(url, stream=True) as r:
    
        # check header to get content length, in bytes
        total_length = int(r.headers.get("Content-Length"))
        
        # implement progress bar via tqdm
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc="")as raw:
        
            # save the output to a file
            with open(path, 'wb')as output:
                shutil.copyfileobj(raw, output)
    
    return path
