import torch
import torch.nn as nn
import torch.nn.functional as F


def set_start_epoch(cfg, overwrite=-1):
    if overwrite>0:
        return overwrite
    if cfg.student_loadmodel:
        cfg.loadmodel = cfg.student_loadmodel
    
    if cfg.loadmodel != None:
        if cfg.finetune is None:
            ans = 0 if cfg.loadmodel is None else int(
            cfg.loadmodel.split('/')[-1].split('_')[-2])+1
        elif cfg.finetune in cfg.loadmodel:
            ans = 0 if cfg.loadmodel is None else int(
            cfg.loadmodel.split('/')[-1].split('_')[-2])+1
        else:
            ans = 0
    else:
        ans =0
    return ans


def pad_size(imgL, imgR, base=32):
    top_pad = (imgL.shape[2]//base+1)*base-imgL.shape[2]
    right_pad = (imgL.shape[3]//base+1)*base-imgL.shape[3]

    imgL = F.pad(
        imgL, (0, right_pad, top_pad, 0),
        "constant", 0
    )
    imgR = F.pad(
        imgR, (0, right_pad, top_pad, 0),
        "constant", 0
    )
    return imgL, imgR, top_pad, right_pad

# def get_save_dirs(cfg, root='.'):
#     if cfg.teacher_loadmodel:
        
#     save=  "./zoo_student/{}".format(cfg.finetune)
def init_cfg(cfg):
    cfg.max_epoch = 3000
    cfg.disp_batch = 1
    cfg.use_cuda = '0'
    cfg.head_only = False
    cfg.freeze_head = False
    # cfg.dist_url = "tcp://127.0.0.1:{}".format(8000 % 65535)
    cfg.dist_url = "tcp://127.0.0.1:{}".format(51593 % 65535)
    cfg.finetune = None
    cfg.loadmodel = None
    cfg.teacher_loadmodel = None
    cfg.student_loadmodel = None
    return cfg

def get_cfg(cfg):
    # if cfg.finetune is None:
    # Sceneflow max size: (540, 960)
    cfg.want_size = (480, 672)
    cfg.disp_batch = 5
    cfg.max_epoch = 600
    cfg.LR_start = 100
    cfg.LR_base = 30
    if cfg.finetune == 'driving':
        # DrivingStereo max size: (400, 881)
        cfg.want_size = (320, 704)
        cfg.disp_batch = 8
        cfg.max_epoch = 6000
        cfg.LR_start = 20
        cfg.LR_base = 5
    if cfg.finetune == 'kitti':
        # KITTI max size: (375, 1242)
        cfg.want_size = (320, 640)
        cfg.disp_batch = 8
        cfg.max_epoch = 6000
        cfg.LR_start = 1000
        cfg.LR_base = 400
    print("***************************************************************************")
    print("***                         SceneFlow Fly                               ***")
    print("***          max epoch: {}, want size: {}, disp batch: {}      ***".format(cfg.max_epoch,cfg.want_size,cfg.disp_batch))
    print("***                                                                     ***")
    print("***************************************************************************")
    cfg.gpu_num = len(cfg.use_cuda.split(','))
    cfg.start_epoch = set_start_epoch(cfg)
    return cfg