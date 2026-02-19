from torch.utils.data import DataLoader
from ImagesCameras import ImageTensor
from datasets.FLIR import FLIR_reg_day, FLIR_reg_night, FLIR, FLIR_DAY, FLIR_NIGHT, FLIR, FLIR_DAY_SAMPLES, \
    FLIR_NIGHT_SAMPLES, FLIR_video_1, FLIR_video_2, FLIR_video_3, FLIR_video_4, Extract_FLIR, FLIR_test, ROADSCENE
from datasets.LYNRED import LYNRED_DAY, LYNRED_NIGHT, LYNRED_NIGHT_SAMPLES, LYNRED_DAY_SAMPLES, LYNRED_EXP1, \
    LYNRED_EXP2, LYNRED_EXP3, LYNRED_EXP4, LYNRED_NIGHT_TRAIN, LYNRED_EXP5, LYNRED_EXP6, LYNRED_EXP7

DATASETS = {
    'FLIR'.lower(): FLIR,
    'FLIR_test'.lower(): FLIR_test,
    'FLIR_reg_day'.lower(): FLIR_reg_day,
    'FLIR_reg_night'.lower(): FLIR_reg_night,
    'FLIR_day'.lower(): FLIR_DAY,
    'FLIR_night'.lower(): FLIR_NIGHT,
    'FLIR_night_SAMPLES'.lower(): FLIR_NIGHT_SAMPLES,
    'FLIR_day_SAMPLES'.lower(): FLIR_DAY_SAMPLES,
    'FLIR_video_1'.lower(): FLIR_video_1,
    'FLIR_video_2'.lower(): FLIR_video_2,
    'FLIR_video_3'.lower(): FLIR_video_3,
    'FLIR_video_4'.lower(): FLIR_video_4,
    'ROADSCENE'.lower(): ROADSCENE,
    'LYNRED_day'.lower(): LYNRED_DAY,
    'LYNRED_night'.lower(): LYNRED_NIGHT,
    'LYNRED_night_SAMPLES'.lower(): LYNRED_NIGHT_SAMPLES,
    'LYNRED_day_SAMPLES'.lower(): LYNRED_DAY_SAMPLES,
    'exp1'.lower(): LYNRED_EXP1,
    'exp2'.lower(): LYNRED_EXP2,
    'exp3'.lower(): LYNRED_EXP3,
    'exp4'.lower(): LYNRED_EXP4,
    'exp5'.lower(): LYNRED_EXP5,
    'exp6'.lower(): LYNRED_EXP6,
    'exp7'.lower(): LYNRED_EXP7,
    'Extract_FLIR'.lower(): Extract_FLIR,
    'LYNRED_NIGHT_TRAIN'.lower(): LYNRED_NIGHT_TRAIN,

}


def collate_ImageTensor(batch):
    """
    Custom collate function to handle ImageTensor objects in a batch.
    """
    images_vis = [item[0] for item in batch]
    images_ir = [item[1] for item in batch]
    original_sizes = [item[2] for item in batch]
    return ImageTensor.batch(*images_vis), ImageTensor.batch(*images_ir), original_sizes


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
            dataloaders[dataset] = DataLoader(DATASETS[dataset.lower()](opt),
                                              batch_size=1,
                                              shuffle=shuffle,
                                              collate_fn=collate_ImageTensor)
        else:
            raise ValueError(f"Dataset {dataset} is not supported. Choose among: {', '.join(list(DATASETS.keys()))}")
    return dataloaders
