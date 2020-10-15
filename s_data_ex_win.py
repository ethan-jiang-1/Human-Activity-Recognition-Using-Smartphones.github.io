from IPython.display import display
from glob import glob
import os

_Raw_data_paths = None


def _find_raw_data_paths():
    global _Raw_data_paths
    if _Raw_data_paths is None:
        if os.path.isdir("ex_win/dat"):
            _Raw_data_paths = sorted(glob("ex_win/dat/*.txt"))
        elif os.path.isdir("dat"):
            _Raw_data_paths = sorted(glob("dat/*.txt"))
    return _Raw_data_paths


def _import_raw_signal(file_path):
    # Create a list
    signal = []
    with open(file_path, 'r') as f:
        line = f.read()
        if line is not None:
            signal = [float(element) for element in line.split(' ')]
    return signal


#display(Raw_data_paths)
_Raw_dic = {}


def load_data_win():
    global _Raw_dic
    rdps = _find_raw_data_paths()
    if rdps is not None:
        for dp in rdps:
            signal = _import_raw_signal(dp)
            _Raw_dic[os.path.basename(dp)] = signal
    return _Raw_dic, [os.path.basename(dp) for dp in rdps]
