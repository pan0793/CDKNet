import numpy as np
from models import *

# from datasets import 
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from datetime import datetime, date
import os
from utils.train_util import *
from utils.common import init_cfg
os.environ['TORCH_DISTRIBUTED_DEBUG']='DETAIL'


def main_worker(gpu, cfg):
    if len(cfg.use_cuda)>1:
        dist.init_process_group(backend = "nccl", world_size=cfg.gpu_num, init_method=cfg.dist_url, rank=gpu)
    if(main_process(gpu)):
        writer = SummaryWriter('runs/exp/{}'.format(date.today()))

    from models.warpers import KD_warper
    # from models.teacher.modules_save import teacher_main
    # from models.teacher.modules_save import teacher_main
    from models.student.modules import student_main
    from models.naive.modules import naive_main
    torch.cuda.set_device(gpu)
    # teacher = teacher_main(full_shape=cfg.want_size, KL_mode=True)
    teacher = student_main(full_shape=cfg.want_size, KL_mode=True)
    student = naive_main(full_shape=cfg.want_size, KL_mode=True)
    # print(len(student.state_dict()))
    # teacher, student, optimizer = load_model_KD3(
    #     teacher=teacher, student=student, optimizer=None, cfg=cfg, gpu=gpu)
    teacher, student, optimizer = load_model_KD4(
        teacher=teacher, student=student, optimizer=None, cfg=cfg, gpu=gpu)
    # teacher, student, optimizer = load_model_KD_student_naive(
    #     teacher=teacher, student=student, optimizer=None, cfg=cfg, gpu=gpu)
    teacher.eval()
    student.train()
    model = KD_warper(teacher, student,KDlossOnly=cfg.KDlossOnly).cuda()
    if cfg.gpu_num>1:
        print("Let's use GPU: {}!".format(gpu))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu])
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3,
                            betas=(0.9, 0.999), weight_decay=1e-2,)
    optimizer = load_model_optimizer(optimizer, cfg, gpu)
    
    # adjust_learning_rate(optimizer=optimizer, epoch=0)


    # if cfg.teacher_loadmodel:
    #     model, optimizer = load_model(model=model, optimizer=optimizer, cfg=cfg, gpu=gpu)
    # if cfg.student_loadmodel:
    #     model, optimizer = load_model(model=model, optimizer=optimizer, cfg=cfg, gpu=gpu)
        # adjust_learning_rate(optimizer=optimizer, epoch=0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, patience=5, verbose=True, threshold=3e-3, factor=0.5)
    TrainImgLoader_disp, TestImgLoader_disp = DATASET_disp(cfg)
    print(len(TrainImgLoader_disp))
    print(len(TestImgLoader_disp))
    small_test_loss = 100
    for epoch in range(cfg.start_epoch, cfg.max_epoch+1):
        adjust_learning_rate(cfg =cfg, optimizer=optimizer, epoch=epoch)
        TrainImgLoader_disp.sampler.set_epoch(epoch)
        # epoch_loss = []
        start_time = datetime.now()
        # if 0:
        for batch_idx, data_batch in enumerate(TrainImgLoader_disp):
            
            # ! step 1: train disp
            # model.model.train()
            loss, loss_disp, loss_head, loss_kd = train(model, data_batch, gpu, optimizer)
            # ! end 
            # epoch_loss.append(float(loss))
            if main_process(gpu) and (batch_idx % (len(TrainImgLoader_disp)//1) == 0) and not eval_epoch(epoch,cfg):
                message = 'Epoch: {}/{}. Iteration: {}/{}. LR:{:.1e},  Epoch time: {}. Disp loss: {:.3f}. Head loss: {:.3f}. KD loss: {:.3f}. Total loss: {:.3f}. '.format(
                    epoch, cfg.max_epoch, batch_idx, len(TrainImgLoader_disp),                    
                    float(optimizer.param_groups[0]['lr']), str(datetime.now()-start_time)[:-4],
                    loss_disp, loss_head, loss_kd, loss)
                print(message)
                step = batch_idx+epoch*len(TrainImgLoader_disp)
                writer.add_text('train/record', message, epoch)
                writer.add_scalar('train/Loss', loss, step)
                writer.add_scalar('train/Disp_loss',  loss_disp, step)
                writer.add_scalar('train/Head_loss',  loss_head, step)
                writer.add_scalar('train/Knoledge_loss',  loss_kd, step)


        # ! -------------------eval-------------------
        if eval_epoch(epoch, cfg):
            loss_all = []
            start_time = datetime.now()
            for _, data_batch in enumerate(TestImgLoader_disp):
                with torch.no_grad():
                    # model.model.eval()
                    disp_loss, head_loss = test(model, data_batch, gpu)
                if cfg.head_only:
                    loss_all.append(head_loss)
                else:
                    loss_all.append(disp_loss)

            loss_all = np.mean(loss_all, 0)
            loss_all = Error_broadcast(loss_all,cfg.gpu_num)
            if main_process(gpu):
                writer.add_scalar('full test/EPE', loss_all[0], epoch)
                writer.add_scalar('full test/D1', loss_all[1], epoch)
                message = 'Epoch: {}/{}. Epoch time: {}. Eval,  Disp loss: {:.3f}, D1 LOSS: {:.3f}%'.format(
                    epoch, cfg.max_epoch, str(datetime.now()-start_time)[:-4],
                    loss_all[0], loss_all[1]*100)
                print(message)
                writer.add_text('full test/record', message, epoch)
                # small_test_loss = loss_all
                save_model_dict(epoch, model.state_dict(),
                                optimizer.state_dict(), loss_all[0], cfg)

if __name__ == '__main__':
    import argparse
    from utils.common import init_cfg, get_cfg
    parser = argparse.ArgumentParser(description='PSMNet')
    cfg = init_cfg(parser.parse_args())
    cfg.max_epoch = 3000
    # cfg.KDlossOnly=True
    cfg.KDlossOnly=False
    cfg.use_cuda = '0,1'
    # cfg.use_cuda = '0'
    cfg.gpu_num = len(cfg.use_cuda.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.use_cuda
    cfg.server_name = 'local'
    # cfg.want_size = (384, 640)
    # cfg.want_size = (480, 640)
    # cfg.loadmodel = None
    # cfg.finetune = None
    # cfg.loadmodel = 'zoo/driving_55_0.36760.tar'
    # cfg.loadmodel = 'zoo/kitti_258_0.38197.tar'
    # cfg.teacher_loadmodel = 'zoo_test/kitti_278_0.37840.tar'
    # cfg.teacher_loadmodel = 'zoo_keep/test_100_0.58014.tar'
    # cfg.teacher_loadmodel = 'zoo_best/test_noheadloss_170_0.51743.tar'
    # cfg.teacher_loadmodel = 'zoo_student_noheadloss_v2/teacher_student_v11_311_0.83415.tar'
    cfg.teacher_loadmodel = 'zoo_best/teacher_student_v11_driving_57_0.62993.tar'
    
    cfg.student_loadmodel = 'zoo_student_naive/student_naive_v11_driving_25_0.68214.tar'
    # cfg.student_loadmodel = 'zoo_student_noheadloss/teacher_student_169_1.36847.tar'
    # cfg.student_loadmodel = "zoo_teacher_student/teacher_student_v11_311_0.83415.tar"
    # cfg.student_loadmodel = "zoo_student/teacher_student_46_1.09732.tar"
    cfg.finetune = 'driving'
    cfg.save_prefix = "./zoo_student_naive/{}".format("student_naive_v11_driving")
    # cfg.start_epoch = set_start_epoch(cfg)
    cfg = get_cfg(cfg)
    cfg.disp_batch = 26
    if len(cfg.use_cuda)>1:
        import torch.multiprocessing as mp
        mp.spawn(main_worker, nprocs=cfg.gpu_num,
                 args=(cfg,))
    else:
        main_worker(int(cfg.use_cuda), cfg)

