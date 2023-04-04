import torch
import torch.nn as nn
from utils.common import pad_size
import torch.distributed as dist
import numpy as np

# def train(model, data_batch, gpu, optimizer):
#     model.train()
#     def f(a): return a.cuda(gpu) if a is not None else a
#     imgL = f(data_batch['left'])
#     imgR = f(data_batch['right'])
#     disp_L = f(data_batch['disp'])
#     disp_L = torch.squeeze(disp_L, dim=1)
#     optimizer.zero_grad()
#     loss_list = model(imgL, imgR, disp_L)
#     if(len(loss_list) == 2):
#         loss_disp, loss_head = loss_list
#         loss = loss_disp
#     elif(len(loss_list) == 3):
#         loss_disp, loss_head, loss_kd = loss_list
#         loss = loss_disp + loss_kd
#     loss.backward()
#     optimizer.step()
#     # if loss == 0 or not torch.isfinite(loss):
#     #     return 0
#     ans = (
#         loss.item() if not isinstance(loss, int) else loss,
#         loss_disp.item() if not isinstance(loss_disp, int) else loss_disp, 
#         loss_head.item() if not isinstance(loss_head, int) else loss_head,
#     )
#     if len(loss_list) == 3:
#         ans += (loss_kd.item() if not isinstance(loss_kd, float) else loss_kd , )

#     return ans

def train(model, data_batch, gpu, optimizer):
    model.train()
    def f(a): return a.cuda(gpu) if a is not None else a
    imgL = f(data_batch['left'])
    imgR = f(data_batch['right'])
    disp_L = f(data_batch['disp'])
    disp_L = torch.squeeze(disp_L, dim=1)
    optimizer.zero_grad()
    loss_list = model(imgL, imgR, disp_L)
    if(len(loss_list) == 2):
        loss_disp, loss_head = loss_list
        loss = loss_disp
    elif(len(loss_list) == 3):
        loss_disp, loss_head, loss_kd = loss_list
        loss = loss_disp + loss_kd
    loss.backward()
    optimizer.step()

    ans = (
        loss.item() if not isinstance(loss, int) else loss,
        loss_disp.item() if not isinstance(loss_disp, int) else loss_disp, 
        loss_head.item() if not isinstance(loss_head, int) else loss_head,
    )
    if len(loss_list) == 3:
        ans += (loss_kd.item() if not (isinstance(loss_kd, float) or isinstance(loss_kd, int)) else loss_kd , )

    return ans




def test(model, data_batch, gpu):
    model.eval()
    def f(a): return a.cuda(gpu) if a is not None else a
    imgL = f(data_batch['left'])
    imgR = f(data_batch['right'])
    disp_L = f(data_batch['disp'])

    disp_L = torch.squeeze(disp_L, dim=1)
    loss_disp, loss_head = model(imgL, imgR, disp_L)
    
    for idx, l in enumerate(loss_disp):
        if l == 0 or not torch.isfinite(l):
            l = 0
        else:
            l = l.item()
        loss_disp[idx] = l

    # if loss_head == 0 or not torch.isfinite(loss_head):
    #     loss_head = 0
    # else:
    #     loss_head = loss_head.item()
    return loss_disp, 0


def main_process(rank):
    return rank == 0


def DATASET_disp(cfg):
    if cfg.finetune is None:
        print('TRAIN on Sceneflow')
        from datasets.dataset import SceneFlowDataset as DATASET
        # list_filename_train = "filenames/sceneflow_train_fly.txt"
        # list_filename_test = "filenames/sceneflow_test_fly.txt"
        list_filename_train = "filenames/sceneflow_train.txt"
        list_filename_test = "filenames/sceneflow_test.txt"
        Train_Dataset = DATASET(want_size=cfg.want_size, list_filename=list_filename_train,
                            training=True, server_name=cfg.server_name)
        Test_Dataset = DATASET(
            training=False, want_size=(0,0), list_filename=list_filename_test, mode='val',
            server_name=cfg.server_name,
        )
        # Train_Dataset = DATASET(want_size=(480, 640),
        #                         training=True, list_filename="filenames/sceneflow_train.txt", server_name=cfg.server_name)
        # Test_Dataset = DATASET(
        #     training=False, want_size=cfg.pad_size, list_filename="filenames/sceneflow_test.txt", server_name=cfg.server_name)
    
    if cfg.finetune == 'driving':
        print('TRAIN on Driving Stereo')
        from datasets.DrivingStereo_loader import Drivingstereo as DATASET
        list_filename_train = "filenames/driving_train.txt"
        list_filename_test = "filenames/driving_test.txt"
        Train_Dataset = DATASET(want_size=cfg.want_size, list_filename=list_filename_train,
                            training=True, server_name=cfg.server_name)
        Test_Dataset = DATASET(
            training=False, want_size=(0,0), list_filename=list_filename_test, mode='val',
            server_name=cfg.server_name,
        )
        # Train_Dataset = DATASET(want_size=cfg.want_size,
        #                         training=True, server_name=cfg.server_name)
        # Test_Dataset = DATASET(
        #     training=False, want_size=cfg.pad_size, mode='val',
        #     server_name=cfg.server_name
        #     )
        
    if cfg.finetune == 'kitti':
        print('TRAIN on KITTI')
        # from datasets.dataset import KITTIDataset as DATASET
        from datasets.dataset import KITTIDataset_1215 as DATASET
        list_filename_train = "filenames/kitti_12_15_train.txt"
        list_filename_test = "filenames/kitti15_test.txt"
        Train_Dataset = DATASET(want_size=cfg.want_size,dataset='12', list_filename=list_filename_train,
                            training=True, server_name=cfg.server_name)
        Test_Dataset = DATASET(
            training=False, want_size=(0,0),dataset='12', list_filename=list_filename_test, mode='val',
            server_name=cfg.server_name,
        )
        
    if len(cfg.use_cuda) > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            Train_Dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            Test_Dataset)
    else:
        train_sampler, test_sampler = None, None

    TrainImgLoader = torch.utils.data.DataLoader(
        Train_Dataset,
        batch_size=cfg.disp_batch,
        shuffle=(train_sampler is None),
        num_workers=2,
        drop_last=True,
        collate_fn=BatchCollator(cfg),
        sampler=train_sampler
    )

    TestImgLoader = torch.utils.data.DataLoader(
        Test_Dataset,
        batch_size=1,
        shuffle=False,
        num_workers=12,
        drop_last=False,
        collate_fn=BatchCollator(cfg),
        sampler=test_sampler
    )
    return TrainImgLoader, TestImgLoader


def load_model(model, optimizer, cfg, gpu):
    print('load model ' + cfg.loadmodel)
    state_dict = torch.load(
        cfg.loadmodel, map_location='cuda:{}'.format(gpu))
    # if args.distributed:
    # update = True
    model_dict = load_model_statedict(model.state_dict(), state_dict['state_dict'], cfg.gpu_num, update=True)
    model.load_state_dict(model_dict)
    
    if optimizer:
        optimizer.load_state_dict(state_dict['optimizer'])
    if cfg.gpu_num > 1:
        dist.barrier()
    return model, optimizer


def load_model_KD(model, optimizer, cfg, gpu):
    print('load model ' + cfg.teacher_loadmodel)
    teacher_state_dict = torch.load(cfg.teacher_loadmodel, map_location='cuda:{}'.format(gpu))['state_dict']
    teacher_state_dict = {k.replace(".model.",".T_model."): v for k,
                               v in teacher_state_dict.items()}
    
    model_dict = load_model_statedict(model.state_dict(), teacher_state_dict, cfg.gpu_num, update=True)
    
    
    if cfg.student_loadmodel:
        student_state_dict = torch.load(
            cfg.student_loadmodel, map_location='cuda:{}'.format(gpu))
        model_dict2 = load_model_statedict(model.state_dict(), student_state_dict, cfg.gpu_num, update=True)
        model_dict.update(model_dict2)
        if optimizer:
            optimizer.load_state_dict(student_state_dict['optimizer'])
    model.load_state_dict(model_dict)
    
    if cfg.gpu_num > 1:
        dist.barrier()
    return model, optimizer

def load_model_KD3(teacher, student, optimizer, cfg, gpu):
    print('load teacher model ' + cfg.teacher_loadmodel)
    teacher_state_dict = torch.load(cfg.teacher_loadmodel, map_location='cuda:{}'.format(gpu))['state_dict']
    teacher_state_dict = {k.replace("model.",""): v for k,
                               v in teacher_state_dict.items()}
    
    teacher_state_dict = load_model_statedict2(teacher.state_dict(), teacher_state_dict, cfg.gpu_num, update=True)
    teacher.load_state_dict(teacher_state_dict)
    
    if cfg.student_loadmodel:
        print('load student model: {}'.format(cfg.student_loadmodel))
        student_state_dict = torch.load(
            cfg.student_loadmodel, map_location='cuda:{}'.format(gpu))['state_dict']
        student_state_dict = {k: v for k,
                               v in student_state_dict.items() if "T_model" not in k}
        print("student dict len: {}".format(len(student_state_dict)))
        student_state_dict = load_model_statedict2(student.state_dict(), student_state_dict, cfg.gpu_num, update=True)
        student.load_state_dict(student_state_dict)
        # if optimizer:
        #     optimizer.load_state_dict(student_state_dict['optimizer'])
        
    if cfg.gpu_num > 1:
        dist.barrier()
    return teacher, student, optimizer

def load_model_KD4(teacher, student, optimizer, cfg, gpu):
    print('load teacher model ' + cfg.teacher_loadmodel)
    teacher_state_dict = torch.load(cfg.teacher_loadmodel, map_location='cuda:{}'.format(gpu))['state_dict']
    teacher_state_dict = {k.replace("model.",""): v for k,
                               v in teacher_state_dict.items() if 'T_model' not in k}
    
    teacher_state_dict = load_model_statedict2(teacher.state_dict(), teacher_state_dict, cfg.gpu_num, update=True)
    teacher.load_state_dict(teacher_state_dict)
    
    if cfg.student_loadmodel:
        print('load student model: {}'.format(cfg.student_loadmodel))
        student_state_dict = torch.load(
            cfg.student_loadmodel, map_location='cuda:{}'.format(gpu))['state_dict']
        student_state_dict = {k.replace("model.",""): v for k,
                               v in student_state_dict.items() if "T_model" not in k}
        print("student dict len: {}".format(len(student_state_dict)))
        student_state_dict = load_model_statedict2(student.state_dict(), student_state_dict, cfg.gpu_num, update=True)
        student.load_state_dict(student_state_dict)
        # if optimizer:
        #     optimizer.load_state_dict(student_state_dict['optimizer'])
        
    if cfg.gpu_num > 1:
        dist.barrier()
    return teacher, student, optimizer


def load_model_optimizer(optimizer, cfg, gpu):
    if cfg.student_loadmodel:
        print('optimizer restored')
        optimizer_state_dict = torch.load(
                cfg.student_loadmodel, map_location='cuda:{}'.format(gpu))
        optimizer.load_state_dict(optimizer_state_dict['optimizer'])
    return optimizer


def load_model_KD2(model, optimizer, cfg, gpu):
    model_dict={}
    if cfg.teacher_loadmodel:
        print('load model ' + cfg.teacher_loadmodel)
        teacher_state_dict = torch.load(cfg.teacher_loadmodel, map_location='cuda:{}'.format(gpu))['state_dict']
        teacher_state_dict = {k.replace(".model.",".T_model."): v for k,
                                v in teacher_state_dict.items()}
        model_dict.update(load_model_statedict(
            model.state_dict(), teacher_state_dict, cfg.gpu_num, update=True))
        
    if cfg.student_loadmodel:
        student_state_dict = torch.load(
            cfg.student_loadmodel, map_location='cuda:{}'.format(gpu))
        model_dict.update(load_model_statedict(
            model.state_dict(), student_state_dict, cfg.gpu_num, update=True))
        if optimizer:
            optimizer.load_state_dict(student_state_dict['optimizer'])
    model.load_state_dict(model_dict)
    
    if cfg.gpu_num > 1:
        dist.barrier()
    return model, optimizer

def load_model_after_KD(student, optimizer, cfg, gpu):
    print('load model ' + cfg.loadmodel)
    state_dict = torch.load(cfg.loadmodel, map_location='cuda:{}'.format(gpu))['state_dict']
    state_dict = {k: v for k,
                  v in state_dict.items() if 'T_model' not in k}
    # state_dict = {k.replace("model.",""): v for k,
    #                            v in state_dict.items()}
    
    state_dict = load_model_statedict(student.state_dict(), state_dict, cfg.gpu_num, update=True)
    student.load_state_dict(state_dict)

    if cfg.gpu_num > 1:
        dist.barrier()
    return student, None




def load_model_statedict(model_dict, pretrained_dict, gpu_num, update=True):
    if update is True:
        # model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        # pretrained_dict = state_dict['state_dict']
        # pretrained_dict = {k.replace("model.",""): v for k,
        #                        v in pretrained_dict.items()}

        if gpu_num > 1:
            concur = [k for k,
                v in pretrained_dict.items() if 'module' in k]
            if len(concur)>0:
                updated_dict = {k: v for k,
                                v in pretrained_dict.items() if k in model_dict}
            else:
                updated_dict = {'module.'+k: v for k,
                                v in pretrained_dict.items() if 'module.'+k in model_dict}
                
        else:
            # a  = pretrained_dict.keys()
            concur = [k for k,
                      v in pretrained_dict.items() if 'module' in k]
            if len(concur)>0:
                updated_dict = {k[7:]: v for k,
                                v in pretrained_dict.items() if k[7:] in model_dict}
            else:
                updated_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
        assert len(updated_dict) == len(
            pretrained_dict), 'Model weights are not imported properly'
        # 2. overwrite entries in the existing state dict
        model_dict.update(updated_dict)
        # 加载我们真正需要的state_dict
        return model_dict
        model.load_state_dict(model_dict)
    else:
        # model.load_state_dict(pretrained_dict)
        return pretrained_dict
    # return model

def load_model_statedict2(model_dict, pretrained_dict, gpu_num, update=True):
    if update is True:
        # model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        # pretrained_dict = state_dict['state_dict']
        # pretrained_dict = {k.replace("model.",""): v for k,
        #                        v in pretrained_dict.items()}

        updated_dict = {k[7:]: v for k,
                            v in pretrained_dict.items() if k[7:] in model_dict}
        assert len(updated_dict) == len(
            pretrained_dict), 'Model weights are not imported properly'
        # 2. overwrite entries in the existing state dict
        model_dict.update(updated_dict)
        # 加载我们真正需要的state_dict
        return model_dict
        model.load_state_dict(model_dict)
    else:
        # model.load_state_dict(pretrained_dict)
        return pretrained_dict



class BatchCollator(object):
    def __init__(self, cfg):
        super(BatchCollator, self).__init__()
        self.cfg = cfg

    def __call__(self, batch):
        # batch = []
        transpose_batch = list(zip(*batch))
        ret = dict()
        ret['left'] = torch.stack(transpose_batch[0], dim=0)
        ret['right'] = torch.stack(transpose_batch[1], dim=0)
        ret['disp'] = torch.stack(transpose_batch[2], dim=0)
        return ret


def Error_broadcast(loss, cuda_count):
    if cuda_count > 1:
        tensor_list = [torch.cuda.FloatTensor(len(loss)).zero_()
                       for _ in range(cuda_count)]
        myvalue = torch.cuda.FloatTensor(loss)
        dist.all_gather(tensor_list, myvalue)
        losses = [0]*len(loss)
        for t in tensor_list:
            for i in range(len(loss)):
                losses[i] += t[i].item()
        return [_/len(tensor_list) for _ in losses]
    else:
        # you do noting
        return loss


def save_model_dict(epoch, model_state_dict, optimizer_state_dict, loss, cfg):
    # cfg.save_prefix = "./zoo_{}/test_"
    # if cfg.finetune is not None:
    #     savefilename = '{}_{}_{:.5f}.tar'.format(cfg.save_prefix, cfg.finetune, epoch, loss)
    #     # savefilename = './zoo_{}/test_{}_{:.5f}.tar'.format(cfg.finetune, epoch, loss)
    # else:
        # savefilename = './zoo/test_{}_{:.5f}.tar'.format(epoch, loss)
    savefilename = '{}_{}_{:.5f}.tar'.format(cfg.save_prefix, epoch, loss)
    torch.save({
        'state_dict': model_state_dict,
        'optimizer': optimizer_state_dict,
        'test_loss': loss,
    }, savefilename)


def adjust_learning_rate(optimizer, epoch, cfg=None, step=None, args=None):
    lr = 1e-3
    if epoch > cfg.LR_start:
        lr /= pow(2, (epoch - cfg.LR_start) // cfg.LR_base)



    # print('adjust LR to {}'.format(lr))
        
        
    # lr = 1e-3 / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




def eval_epoch(epoch, cfg):
    if cfg.finetune is None:
        return epoch>5
    if cfg.finetune == 'driving':
        return epoch>5
    if cfg.finetune == 'kitti':
        return (epoch > 800 and epoch % 2 == 0) or (epoch > 1000)
