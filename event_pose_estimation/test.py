import argparse
import sys
import time
import random
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from geometry import batch_compute_similarity_transform_torch
sys.path.append('../')
from model import *
import event_pose_estimation as models
from dataloader import CompleDataloader
from loss_funcs import compute_mpjpe, compute_pa_mpjpe, compute_pelvis_mpjpe, \
    compute_pck, compute_pck_head, compute_pck_torso
import collections
import numpy as np
import utils as util
import logging


def test(config):
    # GPU or CPU configuration
    start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    writer = SummaryWriter('%s/%s/%s' % (config['exper']['result_dir'], config['exper']['log_dir'], start_time))
    log_file = '%s/%s/%s/training_log.txt' % (config['exper']['result_dir'], config['exper']['log_dir'], start_time)
    logger = setup_logger(log_file)
    print_args(config, logger)
    logger.info('[tensorboard] %s/%s/%s' % (config['exper']['log_dir'], config['exper']['log_dir'], start_time))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % config['exper']['gpu_id'] if use_cuda else "cpu")
    dtype = torch.float32

    dataset_val = CompleDataloader(
        config=config,
        data_dir=config['data']['data_dir'],
        max_steps=config['data']['max_steps'],
        num_steps=config['data']['num_steps'],
        skip=config['data']['skip'],
        events_input_channel=config['model']['param']['events_input_channel'],
        img_size=config['data']['img_size'],
        bbox_resize=config['data']['bbox_resize'],
        mode='val',
        use_voxel=config['exper']['use_voxel'],
        split=config['data']['split'],
        logger=logger
    )
    val_generator = DataLoader(
        dataset_val,
        batch_size=config['exper']['batch_size'],
        shuffle=False,
        num_workers=config['exper']['num_worker'],
        pin_memory=config['exper']['pin_memory']
    )

    total_iters_val = len(dataset_val) // config['exper']['batch_size'] + 1
    logger.info(f'total_iters: {total_iters_val}')
    smpl_dir = config['model']['param']['smpl_dir']
    logger.info('[smpl_dir] %s' % smpl_dir)

    # set model
    Model = getattr(models, config['model']['name'])
    model = Model(**config['model']['param'], logger=logger)
    # model = ComplNet(
    #     events_input_channel=config['model']['events_input_channel'],
    #     smpl_dir=smpl_dir,
    #     pose_dim=24 * 6,
    #     resnet=config['model']['resnet'],
    #     logger=logger
    # )

    if config['exper']['model_dir']:
        logger.info('[model dir] model loaded from %s' % config['exper']['model_dir'])
        checkpoint = torch.load(config['exper']['model_dir'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    model.eval()  # dropout layers will not work in eval mode
    results = collections.defaultdict(list)
    results['faces'] = model.smpl.faces
    with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
        for iter, data in enumerate(val_generator):
            # data: {events, flows, init_shape, theta, tran, joints2d, joints3d, info}
            for k in data.keys():
                if k != 'info':
                    data[k] = data[k].to(device=device, dtype=dtype)
            print(iter)
            out = model(data)
            B, T = data['events'].size()[0], data['events'].size()[1]
            mpjpe = compute_mpjpe(out['joints3d'].detach(), data['joints3d'])  # [B, T, 24]
            pa_mpjpe = compute_pa_mpjpe(out['joints3d'].detach(), data['joints3d'])  # [B, T, 24]
            pel_mpjpe = compute_pelvis_mpjpe(out['joints3d'].detach(), data['joints3d'])  # [B, T, 24]
            pck = compute_pck(out['joints3d'].detach(), data['joints3d'])
            pck_head = compute_pck_head(out['joints3d'], data['joints3d'])  # [B, T, 24]
            pck_torso = compute_pck_torso(out['joints3d'], data['joints3d'])  # [B, T, 24]
            _, s, R, t = batch_compute_similarity_transform_torch(
                out['joints3d'].view(-1, 24, 3), data['joints3d'].view(-1, 24, 3), True)
            # print(s.size(), R.size(), t.size())
            s = s.unsqueeze(-1).unsqueeze(-1)
            pa_verts = s * R.bmm(out['verts'].reshape(B * T, 6890, 3).permute(0, 2, 1)) + t
            pa_verts = pa_verts.permute(0, 2, 1).view(B, T, 6890, 3)

            target_verts, target_joints3d, _ = model.smpl(
                beta=data['shape'].view(-1, 10),
                theta=data['theta'].view(-1, 72),
                get_skin=True)
            target_verts = target_verts.view(B, T, target_verts.size(1), target_verts.size(2)) + data['tran']
            pve = torch.mean(torch.sqrt(torch.sum((target_verts - pa_verts) ** 2, dim=-1)),
                             dim=-1)  # [B, T]

            # collect results
            results['scalar/mpjpe'].append(torch.mean(mpjpe.detach()))
            results['scalar/pa_mpjpe'].append(torch.mean(pa_mpjpe.detach()))
            results['scalar/pel_mpjpe'].append(torch.mean(pel_mpjpe.detach()))
            results['scalar/pck'].append(pck.detach().float())
            results['scalar/pve'].append(pve.detach())
            results['scalar/pck_head'].append(pck_head.detach().float())
            results['scalar/pck_torso'].append(pck_torso.detach().float())
            # if iter > 10:
            #     break
            # if iter % 2 == 0:
            display = 10
            if iter != 0 and iter % (total_iters_val // display) == 0:
                # print(100 * (epoch + 1) + iter // 1000)
                results['info'] = (data['info'][0][0], data['info'][1][0])
                results['verts'] = out['verts'][0].detach()
                results['gt_verts'] = target_verts[0].detach()
                results['events'] = data['events'][0].detach()
                results['cam_intr'] = data['cam_intr'][0].detach()
                results['grays'] = data['grays_o'][0].detach()
                results['scene'] = data['scene'][0].detach()
                write_tensorboard(writer, results, 0, iter // 100, 'test', config)


        results['mpjpe'] = torch.mean(torch.stack(results['scalar/mpjpe'], dim=0))
        results['pa_mpjpe'] = torch.mean(torch.stack(results['scalar/pa_mpjpe'], dim=0))
        results['pel_mpjpe'] = torch.mean(torch.stack(results['scalar/pel_mpjpe'], dim=0))
        results['pck_head'] = torch.mean(torch.cat(results['scalar/pck_head'], dim=0), dim=(0, 1, 2))
        results['pck_torso'] = torch.mean(torch.cat(results['scalar/pck_torso'], dim=0), dim=(0, 1, 2))
        results['pck'] = torch.mean(torch.cat(results['scalar/pck'], dim=0), dim=(0, 1, 2))
        results['pve'] = torch.mean(torch.cat(results['scalar/pve'], dim=0))

        logger.info(
            '    mpjpe {}\n'
            '    pa_mpjpe {}\n'
            '    pel_mpjpe {}\n'
            '    pck {}\n'
            '    pck_head {}\n'
            '    pck_torso {}\n'
            '    pve {}\n'
            .format(1000 * results['mpjpe'], 1000 * results['pa_mpjpe'],
                    1000 * results['pel_mpjpe'], 100 * results['pck'], 100 * results['pck_head'],
                  100 * results['pck_torso'], 1000 * results['pve']))
    writer.close()


def write_tensorboard(writer, results, epoch, progress, mode, config):
    action, sample_frames_idx = results['info']
    verts = results['verts'].cpu().numpy()
    gt_verts = results['gt_verts'].cpu().numpy()
    ef = results['events'].cpu().numpy()
    grays = results['grays'].cpu().numpy()
    scene = results['scene'].cpu().numpy()
    sample_frames_idx = sample_frames_idx[1:]
    cam_intr = results['cam_intr'].cpu().numpy()
    faces = results['faces']
    fullpics, render_imgs, gt_render_imgs = [], [], []
    for i, frame_idx in enumerate(sample_frames_idx):
        gray = grays[i,...]
        vert = verts[i]
        gt_vert = gt_verts[i]

        render_img = util.render_model(vert, faces, cam_intr[i], np.zeros([3]),
                                       (config['data']['bbox_resize'], config['data']['bbox_resize']), img=gray)

        gt_render_img = util.render_model(gt_vert, faces, cam_intr[i], np.zeros([3]),
                                       (config['data']['bbox_resize'], config['data']['bbox_resize']), img=gray)

        render_imgs.append(render_img)
        gt_render_imgs.append(gt_render_img)

    render_imgs = np.transpose(np.stack(render_imgs, axis=0), [0, 3, 1, 2])
    gt_render_imgs = np.transpose(np.stack(gt_render_imgs, axis=0), [0, 3, 1, 2])

    cond0 = "Normal" if int(scene[0]) == 0 else "blur"

    s1 = int(scene[1])
    if s1 == 0:
        cond1 = "Normal"
    elif s1 == 1:
        cond1 = "Overexposure"
    elif s1 == 2:
        cond1 = "Underexposure"
    else:
        cond1 = "Unknown"
    s2 = int(scene[2])
    if s2 == 0:
        cond2 = "Normal"
    elif s2 == 1:
        cond2 = "Overflow"
    elif s2 == 2:
        cond2 = "Underexposure"
    else:
        cond2 = "Unknown"
    # tagf = '%s/fullpic_%s_%s_%06i-%i' % (mode, cond0, cond1, epoch * 100 + progress, i)
    # tage = '%s/ef_%s_%s_%06i-%i' % (mode, cond0, cond1, epoch * 100 + progress, i)
    tagc = '%s/com_%s_%s_%s_%06i' % (mode, cond2, cond0, cond1, epoch * 100 + progress)
    # tags = '%s/shape_%s_%s_%06i-%i' % (mode, cond0, cond1, epoch * 100 + progress, i)
    # writer.add_image(tagf, fullpics[i], global_step=epoch * 100 + progress, dataformats='HWC')
    # writer.add_image(tage, ef[i, 4:7, ...], global_step=epoch * 100 + progress, dataformats='CHW')
    # writer.add_image(tags, render_imgs[i]/255.0, global_step=epoch * 100 + progress, dataformats='CHW')
    grays = np.transpose(grays, (0, 3, 1, 2))
    com = np.concatenate((ef[:, 4:7, ...], grays / 255.0, render_imgs / 255.0, gt_render_imgs / 255.0), axis=3)
    writer.add_image(tagc, com, global_step=epoch * 100 + progress, dataformats='NCHW')


def print_args(config, logger):
    """ Prints the argparse argmuments applied
    Args:
      args = parser.parse_args()
    """
    max_length = max([len(k) for k, _ in config.items()])
    for k, v in config.items():
        logger.info(' ' * (max_length - len(k)) + k + ': ' + str(v))


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get_config():
    # warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--config', type=str, default='./test_.yaml')
    parser.add_argument('--config_merge', type=str, default='')
    args = parser.parse_args()
    config = util.ConfigParser(args.config)

    if args.config_merge != '':
        config.merge_configs(args.config_merge)
    config = config.config

    return config
def set_seed(seed):
    # Python built-in random seed
    random.seed(seed)
    # NumPy random seed
    np.random.seed(seed)
    # PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    config = get_config()
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    test(config)


if __name__ == '__main__':
    
    set_seed(42)
    main()
