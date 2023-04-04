import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image

class Submission_warper(nn.Module):
    def __init__(self, model=None, maxdisp=192):
        super().__init__()
        self.maxdisp = maxdisp
        self.model = model

    def forward(self, L, R):
        L, R, bot_pad, right_pad = self.pad_img(L, R)
        output = self.model(L, R)
        output = self.unpad_img(output, bot_pad, right_pad)[0]
        return output

    def pad_img(self, L, R, base=32):
        if self.training is True:
            self.model.Regression.set_full_shape((L.shape[2], L.shape[3]))
            return L, R, 0, 0
        else:
            bot_pad = int(
                base-L.shape[2] % base) if int(base-L.shape[2] % base) != base else 0
            right_pad = int(
                base-L.shape[3] % base) if int(base-L.shape[3] % base) != base else 0
            self.model.Regression.set_full_shape(
                (L.shape[2]+bot_pad, L.shape[3]+right_pad))
            L = F.pad(
                L, (0, right_pad, 0, bot_pad),
                "constant", 0
            )
            R = F.pad(
                R, (0, right_pad, 0, bot_pad,),
                "constant", 0
            )
            return L, R, bot_pad, right_pad

    def unpad_img(self, output, bot_pad, right_pad):
        if self.training is True:
            return output
        else:
            output = list(output)
            for i in range(len(output)):
                if output[i] is not None:
                    output[i] = output[i][:, :-bot_pad,
                                          :] if bot_pad > 0 else output[i]
                    output[i] = output[i][:, :, :-
                                          right_pad] if right_pad > 0 else output[i]

            return output
        
        
def KITTTI_submission_dataset():
    from datasets.dataset import KITTIDataset_1215 as DATASET

    Test_Dataset = DATASET(dataset='15',
        training=False, want_size=(320, 480), mode='test',
        server_name='local',
    )
    TestImgLoader = torch.utils.data.DataLoader(
        Test_Dataset,
        batch_size=1,
        shuffle=False,
        num_workers=12,
        drop_last=False,
    )
    return TestImgLoader


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


def save_image(img, filename, save_dir='submission'):
    img = img.detach().cpu().squeeze(0).numpy()
    img = (img*256).astype('uint16')
    img = Image.fromarray(img)
    img.save(save_dir+'/'+filename.split('/')[-1])



def load_model(model, loadmodel):
    print('load model ')
    state_dict = torch.load(
        loadmodel, map_location='cuda:{}'.format(0))
    # if args.distributed:
    update = True

    if update is True:
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = state_dict['state_dict']

        pretrained_dict = {k[7:]: v for k,
                               v in pretrained_dict.items() if k[7:] in model_dict}
        assert len(pretrained_dict) == len(
            state_dict['state_dict']), 'Model weights are not imported properly'
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 加载我们真正需要的state_dict
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(state_dict['state_dict'])

    return model