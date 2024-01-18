import numpy as np
import pickle
import h5py
import os
import pandas as pd

def load_pickle(filename, **kwargs):
    with open(filename, 'rb') as fin:
        return pickle.load(fin, **kwargs)

def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()

def save_hdf5(dat, filename):
    f1 = h5py.File(filename, "w")
    dset1 = f1.create_dataset("data", data=dat,
                              compression='gzip',
                              compression_opts=9,)
    f1.close()

def load_hdf5(filename):
    with h5py.File(filename, "r") as f:
        a_group_key = list(f.keys())[0]
        return list(f[a_group_key])

def when_created(filename):
    return time.ctime(os.path.getmtime(filename))

def delete_file(list_filename):
    for f in list_filename:
        os.remove(f)
