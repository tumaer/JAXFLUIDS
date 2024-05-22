from typing import Dict
import h5py

def save_dict_to_h5(my_dict: Dict, save_path: str) -> None:
    """Saves the dictionary my_dict to an h5 file 
    specified with save_path. The dictionary structure
    is mirrored in the h5file via groups and datasets.

    :param my_dict: Dictionary with data
    :type my_dict: Dict
    :param save_path: Path to h5 file
    :type save_path: str
    """
    if not save_path.endswith(".h5"):
        save_path = save_path + ".h5"
    with h5py.File(save_path, "w") as h5file:
        save_dict_to_grp(h5file, my_dict)

def save_dict_to_grp(grp: h5py.Dataset, my_dict: Dict) -> None:
    for key in my_dict.keys():
        if isinstance(my_dict[key], dict):
            grp1 = grp.create_group(key)
            save_dict_to_grp(grp1, my_dict[key])
        else:
            grp.create_dataset(name=key, data=my_dict[key])

def load_dict_from_h5(load_path: str) -> Dict:
    """Load the h5file from load_path and stores 
    the contents in a dictionary which is return.

    :param load_path: Path to an h5file
    :type load_path: str
    :return: Dictionary with contents from h5file
    :rtype: Dict
    """
    my_dict = {}
    if not load_path.endswith(".h5"):
        load_path = load_path + ".h5"
    with h5py.File(load_path, "r") as h5file:
        load_from_grp_to_dict(h5file, my_dict)
    return my_dict

def load_from_grp_to_dict(grp: h5py.Dataset, my_dict: Dict) -> None:
    for key in grp.keys():
        if isinstance(grp[key], h5py.Dataset):
            if grp[key].ndim >= 1:
                my_dict[key] = grp[key][:]
            else:
                val = grp[key][()]
                if isinstance(val, bytes):
                    my_dict[key] = val.decode("utf-8")
                else:
                    my_dict[key] = val
        else:
            my_dict[key] = {}
            load_from_grp_to_dict(grp[key], my_dict[key])
