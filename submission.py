import os
from utils.submission_util import *
from tqdm import tqdm
from utils.convert_onnx import *
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # from models.teacher.modules import teacher_main
    from models.naive.modules import naive_main
    # from models.student.modules import student_main
    model = Submission_warper(naive_main(
        full_shape=(480, 640), plain_mode=True).cuda())
    model = load_model(
        model, 'zoo_best/test_noheadloss_naive_kitti_6000_0.49176.tar')
    # model = load_model(model, 'zoo_best_epe/test_noheadloss_kitti_3683_0.43672.tar')
    TestImgLoader = KITTTI_submission_dataset()
    t = tqdm(iter(TestImgLoader))
    for data_batch in t:
        model.eval()
        def f(a): return a.cuda(0) if a is not None else a
        imgL = f(data_batch[0])
        imgR = f(data_batch[1])
        filename = data_batch[2][0]
        # imgL.resize
        # imgL = F.interpolate(imgL, (480, 640))
        # imgR = F.interpolate(imgR, (480, 640))
        # convert_onnx(model, imgL, imgR)

        output = model(imgL, imgR)[0]
        save_image(output, filename)
