import importlib
import inspect

from torch.utils.data import DataLoader
from ImagesCameras import ImageTensor

import pkgutil
import os

__all__ = []

from datasets.DatasetBase import DatasetBase

DATASETS = {}

# Iterate through all modules in this package
for module_info in pkgutil.iter_modules([os.path.dirname(__file__)]):
    name = module_info.name
    if name != "DatasetBase":  # Exclude the base class from the datasets list
        __all__.append(name.lower())
        DATASETS[name.lower()] = f"{__name__}.{name}".replace('datasets', '')


# DATASETS = {
#     'FLIR'.lower(): FLIR,
#     'FLIR_test'.lower(): FLIR_test,
#     'FLIR_reg_day'.lower(): FLIR_reg_day,
#     'FLIR_reg_night'.lower(): FLIR_reg_night,
#     'FLIR_day'.lower(): FLIR_DAY,
#     'FLIR_night'.lower(): FLIR_NIGHT,
#     'FLIR_night_SAMPLES'.lower(): FLIR_NIGHT_SAMPLES,
#     'FLIR_day_SAMPLES'.lower(): FLIR_DAY_SAMPLES,
#     'FLIR_video_1'.lower(): FLIR_video_1,
#     'FLIR_video_2'.lower(): FLIR_video_2,
#     'FLIR_video_3'.lower(): FLIR_video_3,
#     'FLIR_video_4'.lower(): FLIR_video_4,
#     'ROADSCENE'.lower(): ROADSCENE,
#     'LYNRED_day'.lower(): LYNRED_DAY,
#     'LYNRED_night'.lower(): LYNRED_NIGHT,
#     'LYNRED_night_SAMPLES'.lower(): LYNRED_NIGHT_SAMPLES,
#     'LYNRED_day_SAMPLES'.lower(): LYNRED_DAY_SAMPLES,
#     'exp1'.lower(): LYNRED_EXP1,
#     'exp2'.lower(): LYNRED_EXP2,
#     'exp3'.lower(): LYNRED_EXP3,
#     'exp4'.lower(): LYNRED_EXP4,
#     'exp5'.lower(): LYNRED_EXP5,
#     'exp6'.lower(): LYNRED_EXP6,
#     'exp7'.lower(): LYNRED_EXP7,
#     'Extract_FLIR'.lower(): Extract_FLIR,
#     'LYNRED_NIGHT_TRAIN'.lower(): LYNRED_NIGHT_TRAIN,
#     'MSRS'.lower(): MSRS
# }


def collate_ImageTensor(batch):
    """
    Custom collate function to handle ImageTensor objects in a batch.
    """
    images_vis = [item[0] for item in batch]
    images_ir = [item[1] for item in batch]
    original_sizes = [item[2] for item in batch]
    return ImageTensor.batch(*images_vis), ImageTensor.batch(*images_ir), original_sizes


def get_dataset_class(module):
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            obj.__module__ == module.__name__   # defined in this module
            and issubclass(obj, module.DatasetBase)  # subclass of base
            and obj is not module.DatasetBase  # not the base itself
        ):
            return obj

    raise RuntimeError(f"No Dataset subclass found in {module.__name__}")


def get_dataloaders(opt):
    """
    Get dataloaders for the specified datasets.
    """
    datasets = opt.dataset
    shuffle = opt.shuffle
    if not isinstance(datasets, list):
        datasets = [datasets]
    dataloaders = {}
    for dataset in datasets:
        if dataset.lower() in DATASETS:
            module = importlib.import_module(f"{DATASETS[dataset.lower()]}", package=__package__)
            dataset_cls = get_dataset_class(module)
            dataloaders[dataset] = DataLoader(dataset_cls(opt),
                                              batch_size=1,
                                              shuffle=shuffle,
                                              collate_fn=collate_ImageTensor)
        else:
            raise ValueError(f"Dataset {dataset} is not supported. Choose among: {', '.join(list(DATASETS.keys()))}")
    return dataloaders
