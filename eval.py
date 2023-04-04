import os
from utils.train_util import *
from tqdm import tqdm


if __name__ == '__main__':
    import argparse
    from utils.common import init_cfg, get_cfg
    parser = argparse.ArgumentParser(description='PSMNet')
    cfg = init_cfg(parser.parse_args())
    cfg.finetune = None
    cfg.use_cuda = '0'
    cfg.server_name = "local"
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.use_cuda
    cfg = get_cfg(cfg)
    
    
    from models.warpers import Loss_warper
    from models.teacher.modules import teacher_main
    from models.naive.modules import naive_main
    from models.student.modules import student_main
    
    # from models.teacher.modules_save import teacher_main
    # cfg.loadmodel = 'zoo/test_88_0.58180.tar'
    # cfg.loadmodel = 'zoo_best_epe/naive_only_143_0.81855.tar'
    cfg.loadmodel = 'zoo_student_only/student_only_kitti_3470_0.46696.tar'
    
    model = Loss_warper(student_main(full_shape=(544, 960),
                        plain_mode=True).cuda())
    model = load_model(model, None, cfg, 0)[0]
    # model = load_model_after_KD(model, None, cfg, 0)[0]
    cfg.finetune='kitti'
    cfg.want_size = list(cfg.want_size)
    _, TestImgLoader = DATASET_disp(cfg)

    loss_all = []
    from datetime import datetime
    start_time = datetime.now()
    t = tqdm(iter(TestImgLoader))
    for data_batch in t:
        model.eval()
        disp_loss, head_loss = test(model, data_batch, 0)

        loss_all.append(disp_loss)
        t.set_description("EPE: {:.04f}, D1: {:.04}%".format(np.mean(loss_all, 0)[0],np.mean(loss_all, 0)[1]*100))
    loss_all = np.mean(loss_all, 0)

    message = 'Epoch: {}/{}. Epoch time: {}. EVAL  EPE: {:.04f}, D1: {:.04}%" '.format(
        0, cfg.max_epoch, str(datetime.now()-start_time)[:-4],
        loss_all[0],loss_all[1]*100)
    print(message)

        # save_image(output,filename)
        
        
        
        
    