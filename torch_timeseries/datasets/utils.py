

import os.path as osp
import sys
from typing import Optional
from tqdm.auto import tqdm
import bz2
import gzip
import os.path as osp
import sys
import tarfile
import zipfile
import shutil
import requests
import errno
import os
import os.path as osp


def makedirs(path: str):
    r"""Recursive directory creation function."""
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e

def download_url(url: str, folder: str, filename: Optional[str] = None):
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
        print(f'Using existing file {filename}', file=sys.stderr)
        return path

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



def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print(f"Extracting {archive} to {extract_root}")
    extract_archive(archive, extract_root, remove_finished)









def extract_tar(path: str, folder: str, mode: str = 'r:gz', log: bool = True):
    r"""Extracts a tar archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        mode (string, optional): The compression mode. (default: :obj:`"r:gz"`)
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    with tarfile.open(path, mode) as f:
        f.extractall(folder)


def extract_zip(path: str, folder: str, log: bool = True):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def extract_bz2(path: str, folder: str, log: bool = True):
    r"""Extracts a bz2 archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    path = osp.abspath(path)
    with bz2.open(path, 'r') as r:
        with open(osp.join(folder, '.'.join(path.split('.')[:-1])), 'wb') as w:
            w.write(r.read())


def extract_gz(path: str, folder: str, log: bool = True):
    r"""Extracts a gz archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    path = osp.abspath(path)
    with gzip.open(path, 'r') as r:
        with open(osp.join(folder, '.'.join(path.split('.')[:-1])), 'wb') as w:
            w.write(r.read())
