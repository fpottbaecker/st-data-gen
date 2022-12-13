import multiprocessing
import os
from functools import partial
from math import prod

import numpy as np
from h5py import File
from scipy.sparse import bmat
from tqdm.auto import tqdm


def load(path):
    return File(path)


def tree(hdf5, /, indent=""):
    for key, value in hdf5.items():
        print(f"{indent}'{key}':\t{value}")
        if callable(getattr(value, "items", None)):
            tree(value, indent + " |- ")


def _read_slice(file, name, sparse_format, transpose, slice):
    hdf5 = load(file)[name]
    data = hdf5[slice]
    if transpose:
        data = data.T
    return slice, sparse_format(data)


def _get_nproc():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return multiprocessing.cpu_count()


def read_sparse_chunks(hdf5, /, sparse_format, transpose=False, n_proc=None):
    if hdf5.chunks is None:
        return sparse_format(hdf5[()])

    n_proc = n_proc or _get_nproc()
    shape = np.array(hdf5.shape)
    chunk_size = np.array(hdf5.chunks)
    n_slices = np.ceil(shape / chunk_size).astype("int64")
    data = np.full(n_slices, None)
    with multiprocessing.Pool(n_proc) as p:
        mapper = partial(_read_slice, hdf5.file.filename, hdf5.name, sparse_format, transpose)
        chunks = tqdm(p.imap(mapper, hdf5.iter_chunks(), 1), desc=hdf5.name, total=prod(n_slices))
        for coordinate, element in chunks:
            data[tuple(np.array([coordinate[0].start, coordinate[1].start]) // chunk_size)] = element
    if transpose:
        data = data.T
    print(f"{hdf5.name}: stacking chunks...")
    return bmat(data, format=sparse_format.format)
