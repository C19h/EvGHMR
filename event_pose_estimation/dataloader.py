import pickle
import pickle

import joblib
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from flow_net.flowlib import flow_to_image
from utils import *


class TrackingDataloader(Dataset):
    def __init__(
            self,
            data_dir='/home/shihao/data_event',
            max_steps=16,
            num_steps=8,
            skip=2,
            events_input_channel=8,
            img_size=256,
            mode='train',
            use_voxel=1,
            use_bbox=1,
            use_flow=1,
            flow_loss=1,
            split='cross',
            use_flow_rgb=False,
            use_hmr_feats=False,
            use_vibe_init=False,
            use_hmr_init=False
    ):
        self.data_dir = data_dir
        self.events_input_channel = events_input_channel
        self.skip = skip
        self.split = split
        self.max_steps = max_steps
        self.num_steps = num_steps
        self.img_size = img_size
        scale = self.img_size / 1280.
        self.cam_intr = np.array([1679.3, 1679.3, 641, 641]) * scale
        self.use_hmr_feats = use_hmr_feats
        self.use_bbox = use_bbox
        self.use_flow = use_flow
        self.use_flow = flow_loss
        self.use_flow_rgb = use_flow_rgb
        self.use_vibe_init = use_vibe_init
        self.use_hmr_init = use_hmr_init
        self.use_voxel = use_voxel
        self.mode = mode
        if os.path.exists('%s/%s_track%02i%02i.pkl' % (self.data_dir, self.mode, self.num_steps, self.skip)):
            self.all_clips = pickle.load(
                open('%s/%s_track%02i%02i.pkl' % (self.data_dir, self.mode, self.num_steps, self.skip), 'rb'))
        else:
            self.all_clips = self.obtain_all_clips()

        if self.use_vibe_init:
            print('[VIBE init]')
            all_clips = []
            for (action, frame_idx) in self.all_clips:
                if os.path.exists('%s/vibe_results_%02i%02i/%s/fullpic%04i_vibe%02i.pkl' %
                                  (self.data_dir, self.num_steps, self.skip, action, frame_idx, self.num_steps)):
                    all_clips.append((action, frame_idx))
                else:
                    print('[vibe not exist] %s %i' % (action, frame_idx))
            self.all_clips = all_clips

        if self.use_hmr_init:
            print('[hmr init]')
            all_clips = []
            for (action, frame_idx) in self.all_clips:
                if os.path.exists('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, frame_idx)):
                    all_clips.append((action, frame_idx))
                else:
                    print('[hmr not exist] %s %i' % (action, frame_idx))
            self.all_clips = all_clips

        print('[%s] %i clips, track%02i%02i.pkl' % (self.mode, len(self.all_clips), self.num_steps, self.skip))

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, idx):
        action, frame_idx = self.all_clips[idx]
        if self.mode == 'train':
            next_frames_idx = self.skip * np.sort(np.random.choice(
                np.arange(1, self.max_steps + 1), self.num_steps, replace=False))
        else:
            # test
            next_frames_idx = self.skip * np.arange(1, self.num_steps + 1)

        sample_frames_idx = np.append(frame_idx, frame_idx + next_frames_idx)
        # print(sample_frames_idx)

        if self.use_vibe_init:
            _, _, _params, _tran = joblib.load(
                '%s/vibe_results_%02i%02i/%s/fullpic%04i_vibe%02i.pkl' %
                (self.data_dir, self.num_steps, self.skip, action, frame_idx, self.num_steps))
            theta = _params[0:1, 3:75]
            beta = _params[0:1, 75:]
            tran = _tran[0:1, :]
            init_shape = np.concatenate([tran, theta, beta], axis=1)
        elif self.use_hmr_init:
            _, _, _params, _tran, _ = \
                joblib.load('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, frame_idx))
            theta = np.expand_dims(_params[3:75], axis=0)
            beta = np.expand_dims(_params[75:], axis=0)
            tran = _tran
            init_shape = np.concatenate([tran, theta, beta], axis=1)
        else:
            beta, theta, tran, _, _ = joblib.load('%s/pose_events/%s/pose%04i.pkl' % (self.data_dir, action, frame_idx))
            init_shape = np.concatenate([tran, theta, beta], axis=1)

        if self.use_hmr_feats:
            _, _, _, _, hmr_feats = joblib.load(
                '%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, frame_idx))  # [2048]
        else:
            hmr_feats = np.zeros([2048])

        events, flows, grays, flows_rgb, beta_list, theta_list, tran_list, joints2d_list, joints3d_list, cam_intr_list = [], [], [], [], [], [], [], [], [], []
        for i in range(self.num_steps):
            start_idx = sample_frames_idx[i]
            end_idx = sample_frames_idx[i + 1]
            # print('frame %i - %i' % (start_idx, end_idx))

            # single step events frame
            single_events_frame = []
            for j in range(start_idx, end_idx):
                if self.use_voxel:
                    single_events_frame.append(
                        cv2.imread('%s/events_%i/%s/event%04i.png' % (self.data_dir, self.img_size, action, j), -1))
                else:
                    single_events_frame.append(
                        np.load('%s/xtore_%i/%s/event%04i.npy' % (self.data_dir, self.img_size, action, j)))
            single_events_frame = np.concatenate(single_events_frame, axis=2).astype(np.float32)  # [H, W, C]
            # aggregate the events frame to get 8 channel
            if single_events_frame.shape[2] > self.events_input_channel:
                skip = single_events_frame.shape[2] // self.events_input_channel
                idx1 = skip * np.arange(self.events_input_channel)
                idx2 = idx1 + skip
                idx2[-1] = max(idx2[-1], single_events_frame.shape[2])
                single_events_frame = np.stack(
                    [(np.sum(single_events_frame[:, :, c1:c2], axis=2) > 0) for (c1, c2) in zip(idx1, idx2)], axis=2)

            if self.use_flow or self.flow_loss:
                # single step flow
                single_flows = [joblib.load(
                    '%s/pred_flow_events_%i/%s/flow%04i.pkl' % (self.data_dir, self.img_size, action, j))
                    for j in range(start_idx, end_idx, self.skip)]  # flow is predicted with skip=2
                # single flow is saved as int16 to save disk memory, [T, C, H, W]
                single_flows = np.stack(single_flows, axis=0).astype(np.float32) / 100
                single_flows = np.sum(single_flows, axis=0)
                if self.use_flow_rgb:
                    single_flows_rgb = np.transpose(
                        flow_to_image(np.transpose(single_flows, [1, 2, 0])), [2, 0, 1]) / 255.
                else:
                    single_flows_rgb = np.array([0])
            else:
                single_flows = np.array([0])
                single_flows_rgb = np.array([0])

            flows_rgb.append(single_flows_rgb)

            # single frame pose
            beta, theta, tran, joints3d, joints2d = joblib.load(
                '%s/pose_events/%s/pose%04i.pkl' % (self.data_dir, action, end_idx))
            cam_intr = self.cam_intr
            gray = cv2.imread('%s/full_pic_%i/%s/fullpic%04i.jpg' % (self.data_dir, self.img_size, action, end_idx))

            if self.use_bbox:
                x_min = np.min(joints2d[:, 0])
                x_max = np.max(joints2d[:, 0])
                y_min = np.min(joints2d[:, 1])
                y_max = np.max(joints2d[:, 1])
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                bbox_size = np.max([x_max - x_min, y_max - y_min]) * 1.3
                bbox = np.array([center_x, center_y, bbox_size, bbox_size])
                bbox_scale = self.img_size / bbox_size
                single_events_frame = get_single_image_crop(single_events_frame, bbox, scale=1, crop_size=self.img_size)
                gray = get_single_image_crop(gray, bbox, scale=1, crop_size=self.img_size)
                if self.use_flow or self.flow_loss:
                    single_flows = get_single_image_crop(single_flows.transpose(1, 2, 0), bbox, scale=1,
                                                         crop_size=self.img_size)
                    single_flows = single_flows.transpose(2, 0, 1)
                joints2d, cam_intr = adjust_2d_joints_and_camera(bbox, bbox_scale, joints2d[..., :2], cam_intr)
                # pred = projection_np(joints3d, cam_intr)
            grays.append(gray)
            events.append(single_events_frame)
            flows.append(single_flows)
            beta_list.append(beta[0])
            theta_list.append(theta[0])
            tran_list.append(tran[0])
            joints2d_list.append(joints2d)
            joints3d_list.append(joints3d)
            cam_intr_list.append(cam_intr)

        events = np.stack(events, axis=0)  # [T, H, W, 8]
        flows = np.stack(flows, axis=0)  # [T, 2/3, H, W]
        grays = np.stack(grays, axis=0)  # [T, 2/3, H, W]
        theta_list = np.stack(theta_list, axis=0)  # [T, 72]
        beta_list = np.stack(beta_list, axis=0)  # [T, 72]
        cam_intr_list = np.stack(cam_intr_list, axis=0)  # [T, 4]
        tran_list = np.expand_dims(np.stack(tran_list, axis=0), axis=1)  # [T, 1, 3] in meter
        # [T, 24, 2] drop d, normalize to 0-1
        joints2d_list = np.stack(joints2d_list, axis=0) / self.img_size
        joints3d_list = np.stack(joints3d_list, axis=0)  # [T, 24, 3] added trans

        one_sample = {}
        one_sample['events'] = torch.from_numpy(np.transpose(events, [0, 3, 1, 2])).float()  # [T, 8, H, W]
        one_sample['flows'] = torch.from_numpy(flows).float()  # [T, 2, H, W]
        one_sample['grays'] = torch.from_numpy(grays).float()  # [T, 2, H, W]
        one_sample['flows_rgb'] = torch.from_numpy(flows).float()  # [T, 3, H, W]
        one_sample['init_shape'] = torch.from_numpy(init_shape).float()  # [1, 85]
        one_sample['hidden_feats'] = torch.from_numpy(hmr_feats).float()  # [2048]
        one_sample['theta'] = torch.from_numpy(theta_list).float()  # [T, 72]
        one_sample['shape'] = torch.from_numpy(beta_list).float()  # [T, 10]
        one_sample['tran'] = torch.from_numpy(tran_list).float()  # [T, 1, 3]
        one_sample['joints2d'] = torch.from_numpy(joints2d_list).float()  # [T, 24, 2]
        one_sample['joints3d'] = torch.from_numpy(joints3d_list).float()  # [T, 24, 3]
        one_sample['cam_intr'] = torch.from_numpy(cam_intr_list).float()  # [T, 4]
        one_sample['info'] = [action, sample_frames_idx]
        return one_sample

    def obtain_all_clips(self):
        all_clips = []
        pose_events_dir = os.path.join(self.data_dir, "pose_events")
        actions = sorted(os.listdir(pose_events_dir))
        action_names = []

        for action in actions:
            if self.split == 'cross':
                subject = action.split('_')[0]
                if self.mode == 'test' or self.mode == 'val':
                    if subject in ['subject01', 'subject02', 'subject07']:
                        action_names.append(action)
                else:
                    if subject not in ['subject01', 'subject02', 'subject07']:
                        action_names.append(action)
            else:
                time = action.split('_')[-1]
                if self.mode == 'test' or self.mode == 'val':
                    if time in ['time3']:
                        action_names.append(action)
                else:
                    if time in ['time1', 'time2', 'time4']:
                        action_names.append(action)

        for action in action_names:
            if not os.path.exists('%s/pose_events/%s/pose_info.pkl' % (self.data_dir, action)):
                print('[warning] not exsit %s/pose_events/%s/pose_info.pkl' % (self.data_dir, action))
                continue

            frame_indices = joblib.load('%s/pose_events/%s/pose_info.pkl' % (self.data_dir, action))
            for i in range(len(frame_indices) - self.max_steps * self.skip):
                frame_idx = frame_indices[i]
                end_frame_idx = frame_idx + self.max_steps * self.skip
                if not os.path.exists('%s/pred_flow_events_%i/%s/flow%04i.pkl' %
                                      (self.data_dir, self.img_size, action, end_frame_idx)):
                    # print('flow %i not exists for %s-%i' % (end_frame_idx, action, frame_idx))
                    continue
                if not os.path.exists(
                        '%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, frame_idx)):
                    continue
                if end_frame_idx == frame_indices[i + self.max_steps * self.skip]:
                    # action, frame_idx
                    all_clips.append((action, frame_idx))

        pkl_file = os.path.join(self.data_dir,
                                f"{self.mode}_{self.split}_track{self.num_steps:02d}{self.skip:02d}.pkl")

        with open(pkl_file, 'wb') as f:
            pickle.dump(all_clips, f)

        return all_clips

    def visualize(self, idx):
        sample = self.__getitem__(idx)
        action, sample_frames_idx = sample['info']
        events = np.transpose(sample['events'].numpy(), [0, 2, 3, 1])
        flows = np.transpose(sample['flows'].numpy(), [0, 2, 3, 1])
        joints3d = sample['joints3d'].numpy()
        joints2d = sample['joints2d'].numpy() * self.img_size

        # '''
        from event_pose_estimation.SMPL import SMPL
        model_dir = '../smpl_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
        device = torch.device('cpu')
        smpl_male = SMPL(model_dir, 1).to(device)

        import event_pose_estimation.utils as util
        import matplotlib.pyplot as plt
        from flow_net.flowlib import flow_to_image

        for t in range(self.num_steps // 2):
            plt.figure(figsize=(5 * 4, 5 * 4))

            # fullpic
            prev_fullpic = cv2.imread('%s/full_pic_%i/%s/fullpic%04i.jpg' %
                                      (self.data_dir, self.img_size, action, sample_frames_idx[t]))
            curr_fullpic = cv2.imread('%s/full_pic_%i/%s/fullpic%04i.jpg' %
                                      (self.data_dir, self.img_size, action, sample_frames_idx[t + 1]))
            plt.subplot(4, 4, 1)
            plt.imshow(prev_fullpic[:, :, 0], cmap='gray')
            plt.axis('off')
            plt.title('%s-%04i' % (action, sample_frames_idx[t]))

            plt.subplot(4, 4, 2)
            plt.imshow(curr_fullpic[:, :, 0], cmap='gray')
            plt.axis('off')
            plt.title('%s-%04i' % (action, sample_frames_idx[t + 1]))

            for i in range(self.num_steps):
                plt.subplot(4, 4, 3 + i)
                plt.imshow(events[t, :, :, i], cmap='gray')
                plt.axis('off')

            beta = sample['init_shape'][:, -10:]
            theta = sample['theta'][t:t + 1, :]
            verts, _, _ = smpl_male(beta, theta, get_skin=True)
            verts = (verts[0] + sample['tran'][t, :, :]).numpy()

            faces = smpl_male.faces
            dist = np.abs(np.mean(verts, axis=0)[2])
            render_img = (util.render_model(verts, faces, 256, 256, self.cam_intr, np.zeros([3]),
                                            np.zeros([3]), near=0.1, far=20 + dist, img=curr_fullpic) * 255).astype(
                np.uint8)

            plt.subplot(4, 4, 11)
            plt.imshow(render_img)
            plt.axis('off')

            # joint2d and 3d
            img = curr_fullpic.copy()
            proj_joints2d = util.projection(joints3d[t], self.cam_intr, True)
            for point in proj_joints2d.astype(np.int64):
                cv2.circle(img, (point[0], point[1]), 1, (0, 0, 255), -1)
            plt.subplot(4, 4, 12)
            plt.imshow(img)
            plt.axis('off')
            plt.title('joints3d')

            img = curr_fullpic.copy()
            for point in joints2d[t].astype(np.int64):
                cv2.circle(img, (point[0], point[1]), 1, (255, 0, 0), -1)
            plt.subplot(4, 4, 13)
            plt.imshow(img)
            plt.axis('off')
            plt.title('joints2d')

            flow_rgb = flow_to_image(flows[t])
            plt.subplot(4, 4, 14)
            plt.imshow(flow_rgb)
            plt.axis('off')
            plt.title('flow')

            plt.show()
        # '''


class CompleDataloader(Dataset):
    def __init__(self,
                 config='',
                 data_dir='',
                 max_steps=16,
                 num_steps=8,
                 skip=2,
                 events_input_channel=8,
                 img_size=256,
                 bbox_resize=128,
                 mode='train',
                 use_voxel=1,
                 split='cross',
                 logger=None):
        self.config = config
        self.data_dir = data_dir
        self.events_input_channel = events_input_channel
        self.skip = skip
        self.max_steps = max_steps
        self.num_steps = num_steps
        self.img_size = img_size
        scale = self.img_size / 1280.0
        self.cam_intr = np.array([1679.3, 1679.3, 641, 641]) * scale
        self.bbox_resize = bbox_resize
        self.use_voxel = use_voxel
        self.mode = mode
        self.split = split
        self.logger = logger
        # Initialize degraders for event and grayscale inputs
        self.GDegrader = GDegrader(self.config['exper']['augment']['g_photometry'], mode)
        self.EVDegrader = EVDegrader(self.config['exper']['augment']['event_photometry'], mode)
        self.EvcomDegrader = ComDegrader(self.config['exper']['augment']['event_photometry'], mode=mode)
        self.GcomDegrader = ComDegrader(self.config['exper']['augment']['g_photometry'], mode=mode)
        self.gtrans = BaseRgbTransform()
        pkl_file = os.path.join(self.data_dir,
                                f"{self.mode}_{self.split}_track{self.num_steps:02d}{self.skip:02d}.pkl")

        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as f:
                self.all_clips = pickle.load(f)
        else:
            self.all_clips = self.obtain_all_clips()
        if config['data']['vis']:
            self.all_clips = [(a, f) for (a, f) in self.all_clips if a == 'subject01_group1_time3' and f == 1186]

        self.logger.info(f'[{self.mode}] {len(self.all_clips)} clips, track{self.num_steps:02d}{self.skip:02d}.pkl')

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, idx):
        action, frame_idx = self.all_clips[idx]

        if self.mode == 'train':
            next_frames_idx = self.skip * np.sort(np.random.choice(
                np.arange(1, self.max_steps + 1), self.num_steps, replace=False))
        else:
            next_frames_idx = self.skip * np.arange(1, self.num_steps + 1)
        sample_frames_idx = np.append(frame_idx, frame_idx + next_frames_idx)

        events, flows, grays, scene = [], [], [], []
        beta_list, theta_list, tran_list = [], [], []
        joints2d_list, joints3d_list, cam_intr_list = [], [], []
        p_blur = torch.rand(1)
        is_blur = torch.zeros(1)
        if p_blur < self.config['exper']['augment']['g_photometry']['motion_blur']:
            is_blur[0] = 1
        for i in range(self.num_steps):
            start_idx = sample_frames_idx[i]
            end_idx = sample_frames_idx[i + 1]
            single_events_frame = []
            for j in range(start_idx, end_idx):
                if self.use_voxel:
                    event_path = os.path.join(self.data_dir, f"events_{self.img_size}", action, f"event{j:04d}.png")
                    single_events_frame.append(cv2.imread(event_path, -1))
                else:
                    event_path = os.path.join(self.data_dir, f"xtore_{self.img_size}", action, f"event{j:04d}.npy")
                    single_events_frame.append(np.load(event_path))
            single_events_frame = np.concatenate(single_events_frame, axis=2).astype(np.float32)  # [H, W, C]
            gray_path = os.path.join(self.data_dir, f"full_pic_{self.img_size}", action, f"fullpic{end_idx:04d}.jpg")
            gray = cv2.imread(gray_path)

            if self.mode == 'train' or self.mode == 'val':
                if p_blur < self.config['exper']['augment']['g_photometry']['motion_blur']:
                    next_gray_path = os.path.join(self.data_dir, f"full_pic_{self.img_size}", action,
                                                  f"fullpic{end_idx + 1:04d}.jpg")
                    gray_next = cv2.imread(next_gray_path)
                    gray = simulate_motion_blur_from_frames([gray, gray_next], num_interp=5)

            if single_events_frame.shape[2] > self.events_input_channel:
                skip_channel = single_events_frame.shape[2] // self.events_input_channel
                idx1 = skip_channel * np.arange(self.events_input_channel)
                idx2 = idx1 + skip_channel
                idx2[-1] = max(idx2[-1], single_events_frame.shape[2])
                aggregated = [(np.sum(single_events_frame[:, :, c1:c2], axis=2) > 0)
                              for c1, c2 in zip(idx1, idx2)]
                single_events_frame = np.stack(aggregated, axis=2)

            gray = self.GcomDegrader(torch.from_numpy(gray).float())
            single_events_frame = self.EvcomDegrader(single_events_frame)


            flows_list = []
            for j in range(start_idx, end_idx, self.skip):
                flow_path = os.path.join(self.data_dir, f"pred_flow_events_{self.img_size}", action,
                                         f"flow{j:04d}.pkl")
                flow = joblib.load(flow_path)
                flows_list.append(flow)
            single_flows = np.stack(flows_list, axis=0).astype(np.float32) / 100
            single_flows = np.sum(single_flows, axis=0)

            pose_file_end = os.path.join(self.data_dir, "pose_events", action, f"pose{end_idx:04d}.pkl")
            beta, theta, tran, joints3d, joints2d = joblib.load(pose_file_end)
            cam_intr = self.cam_intr.copy()

            x_min, x_max = np.min(joints2d[:, 0]), np.max(joints2d[:, 0])
            y_min, y_max = np.min(joints2d[:, 1]), np.max(joints2d[:, 1])
            center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
            bbox_size = np.max([x_max - x_min, y_max - y_min]) * 1.3
            center_y = center_y - 0.05 * bbox_size
            bbox = np.array([center_x, center_y, bbox_size, bbox_size])
            bbox_scale = self.bbox_resize / bbox_size
            single_events_frame = get_single_image_crop(single_events_frame, bbox, scale=1, crop_size=self.bbox_resize)
            gray = get_single_image_crop(gray, bbox, scale=1, crop_size=self.bbox_resize)

            single_flows = get_single_image_crop(single_flows.transpose(1, 2, 0), bbox, scale=1,
                                                 crop_size=self.bbox_resize)
            single_flows = single_flows.transpose(2, 0, 1)
            joints2d, cam_intr = adjust_2d_joints_and_camera(bbox, bbox_scale, joints2d[:, :2], cam_intr)

            grays.append(gray)
            events.append(single_events_frame)
            flows.append(single_flows)
            beta_list.append(beta[0])
            theta_list.append(theta[0])
            tran_list.append(tran[0])
            joints2d_list.append(joints2d)
            joints3d_list.append(joints3d)
            cam_intr_list.append(cam_intr)

        events = np.stack(events, axis=0)  # [T, H, W, C]
        events, sce = self.EVDegrader(torch.from_numpy(events).float())
        flows = np.stack(flows, axis=0)  # [T, C, H, W]
        grays = np.stack(grays, axis=0)  # [T, H, W, C]
        grays, scg = self.GDegrader(grays)
        sc = torch.cat([is_blur, scg, sce], dim=0)

        if self.config['model']['name'] == 'ComplNetwoDino':
            grays = self.gtrans(grays)
        theta_list = np.stack(theta_list, axis=0)  # [T, 72]
        beta_list = np.stack(beta_list, axis=0)  # [T, 10]
        cam_intr_list = np.stack(cam_intr_list, axis=0)  # [T, 4]
        tran_list = np.expand_dims(np.stack(tran_list, axis=0), axis=1)  # [T, 1, 3]
        joints2d_list = np.stack(joints2d_list, axis=0) / self.bbox_resize
        joints3d_list = np.stack(joints3d_list, axis=0)

        one_sample = {
            'events': events.permute(0,3,1,2),  # [T, C, H, W]
            'flows': torch.from_numpy(flows).float(),
            'grays': torch.from_numpy(grays).float(),
            'grays_o': torch.from_numpy(grays).float(),
            'theta': torch.from_numpy(theta_list).float(),
            'shape': torch.from_numpy(beta_list).float(),
            'tran': torch.from_numpy(tran_list).float(),
            'scene': sc,
            'joints2d': torch.from_numpy(joints2d_list).float(),
            'joints3d': torch.from_numpy(joints3d_list).float(),
            'cam_intr': torch.from_numpy(cam_intr_list).float(),
            'info': [action, sample_frames_idx]
        }
        return one_sample

    def obtain_all_clips(self):
        all_clips = []
        pose_events_dir = os.path.join(self.data_dir, "pose_events")
        actions = sorted(os.listdir(pose_events_dir))
        action_names = []

        for action in actions:
            if self.split == 'cross':
                subject = action.split('_')[0]
                if self.mode == 'test' or self.mode == 'val':
                    if subject in ['subject01', 'subject02', 'subject07']:
                        action_names.append(action)
                else:
                    if subject not in ['subject01', 'subject02', 'subject07']:
                        action_names.append(action)
            else:
                time = action.split('_')[-1]
                if self.mode == 'test' or self.mode == 'val':
                    if time in ['time3']:
                        action_names.append(action)
                else:
                    if time in ['time1', 'time2', 'time4']:
                        action_names.append(action)

        for action in action_names:
            pose_info_path = os.path.join(self.data_dir, "pose_events", action, "pose_info.pkl")
            if not os.path.exists(pose_info_path):
                self.logger.info(f"[warning] not exist {pose_info_path}")
                continue

            frame_indices = joblib.load(pose_info_path)
            for i in range(len(frame_indices) - self.max_steps * self.skip):
                frame_idx = frame_indices[i]
                end_frame_idx = frame_idx + self.max_steps * self.skip
                flow_path = os.path.join(
                    self.data_dir,
                    f"pred_flow_events_{self.img_size}",
                    action,
                    f"flow{end_frame_idx:04d}.pkl"
                )
                if not os.path.exists(flow_path):
                    continue
                if end_frame_idx == frame_indices[i + self.max_steps * self.skip]:
                    all_clips.append((action, frame_idx))

        pkl_file = os.path.join(self.data_dir,
                                f"{self.mode}_{self.split}_track{self.num_steps:02d}{self.skip:02d}.pkl")
        with open(pkl_file, 'wb') as f:
            pickle.dump(all_clips, f)
        return all_clips


class VisDataloader(CompleDataloader):

    def __init__(self,
                 target_action: str,  # Required: action to visualize (e.g., "subject01_group1_time3")
                 idx_range: tuple = None,  # Optional: index range under this action (start, end), default is all
                 config='',
                 data_dir='',
                 max_steps=16,
                 num_steps=8,
                 skip=2,
                 events_input_channel=8,
                 img_size=256,
                 bbox_resize=128,
                 mode='val',  # Use val mode by default to avoid augmentation side effects
                 use_voxel=1,
                 split='cross',
                 logger=None):
        # 1. Initialize the parent class first (loads all samples)
        super().__init__(
            config=config,
            data_dir=data_dir,
            max_steps=max_steps,
            num_steps=num_steps,
            skip=skip,
            events_input_channel=events_input_channel,
            img_size=img_size,
            bbox_resize=bbox_resize,
            mode=mode,
            use_voxel=use_voxel,
            split=split,
            logger=logger
        )

        # 2. First filter: keep only samples from the target action
        self.target_action = target_action
        self.action_clips = [clip for clip in self.all_clips if clip[0] == self.target_action]

        # Validate that the target action has available samples
        if len(self.action_clips) == 0:
            raise ValueError(f"Error: Action '{self.target_action}' 无可用样本！请检查action名称或数据路径")

        # 3. Second filter: slice samples based on idx_range
        self.idx_range = idx_range
        if idx_range is not None:
            start, end = idx_range
            # Validate index range
            if not isinstance(start, int) or not isinstance(end, int):
                raise TypeError(f"idx_range必须是整数元组！当前输入：{idx_range}")
            if start < 0 or end > len(self.action_clips) or start >= end:
                raise ValueError(
                    f"idx_range({start}, {end}) 非法！\n"
                    f"有效范围：start ∈ [0, {len(self.action_clips) - 1}], end ∈ (start, {len(self.action_clips)}]"
                )
            # Slice samples in the given range
            self.range_clips = self.action_clips[start:end]
        else:
            # If no range is given, use all samples under the action
            self.range_clips = self.action_clips

        # 4. Log filtering results
        log_info = (
            f"[ActionRangeDataloader] 筛选结果：\n"
            f"  目标Action: {self.target_action}\n"
            f"  该Action总样本数: {len(self.action_clips)}\n"
            f"  指定索引区间: {self.idx_range if self.idx_range else '全部'}\n"
            f"  最终加载样本数: {len(self.range_clips)}"
        )
        if self.logger is not None:
            self.logger.info(log_info)
        else:
            print(log_info)

    def __len__(self):
        """Override len: return the sample count after filtering by action and index range."""
        return len(self.range_clips)

    def __getitem__(self, local_idx: int):
        """
        Override __getitem__:
        - local_idx: local index after filtering (0 ~ len(self.range_clips)-1)
        - Internally map to the parent's global index and reuse parent loading logic
        """
        # Validate local index bounds
        if local_idx < 0 or local_idx >= len(self.range_clips):
            raise IndexError(
                f"局部索引{local_idx}超出范围！\n"
                f"有效范围：0 ~ {len(self.range_clips) - 1}（共{len(self.range_clips)}个样本）"
            )

        # 1. Get the clip (action, frame_idx) for the current local index
        target_clip = self.range_clips[local_idx]

        # 2. Find its global index in parent all_clips
        global_idx = self.all_clips.index(target_clip)

        # 3. Call parent __getitem__ to load the sample (fully reusing original logic)
        return super().__getitem__(global_idx)

    def get_clip_info(self):
        """Added method: return detailed info of filtered samples for debugging/visualization."""
        return {
            'target_action': self.target_action,
            'idx_range': self.idx_range,
            'total_samples_after_filter': len(self.range_clips),
            'frame_indices': [clip[1] for clip in self.range_clips],  # frame_idx list for all filtered samples
            'clip_list': self.range_clips  # original (action, frame_idx) list
        }
class GDegrader(object):
    def __init__(self, config, mode):
        self.config = config
        if mode == 'train' or mode == 'val':
            self.de = True
        else:
            self.de = False

        self.po = config['oe']['p']
        self.pu = config['ue']['p']

    @staticmethod
    def get_contrast_factor(brightness_factor, k_under=1, k_over=0.5):
        if brightness_factor < 1.0:

            contrast_factor = max(0, 1 - k_under * (1 - brightness_factor))
        else:
            contrast_factor = max(0, 1 - k_over * (brightness_factor - 1))
        return contrast_factor

    @staticmethod
    def cv_color_jitter_custom(image, brightness_range):
        if isinstance(brightness_range, (tuple, list)):
            brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
        else:
            brightness_factor = np.random.uniform(max(0, 1 - brightness_range), 1 + brightness_range)

        contrast_factor = GDegrader.get_contrast_factor(brightness_factor, k_under=1, k_over=0.5)
        img = image.astype(np.float32)
        img = img * brightness_factor
        img = np.clip(img, 0, 255)
        mean = np.mean(img, axis=(1, 2), keepdims=True)
        img = (img - mean) * contrast_factor + mean
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def __call__(self, gray, seed=None):
        scene = torch.zeros(1)
        if not self.de:
            return gray, scene
        else:
            r = torch.rand(1)
            if seed is not None:
                torch.manual_seed(seed)
            if r < self.po:
                gray = GDegrader.cv_color_jitter_custom(gray, self.config['oe']['brightness'])
                scene[0] = 1
            elif r < self.po + self.pu:
                gray = GDegrader.cv_color_jitter_custom(gray, self.config['ue']['brightness'])
                scene[0] = 2
            return gray, scene


class EVDegrader(object):
    def __init__(self, config, mode):
        self.config = config
        if mode == 'train' or mode == 'val':
            self.de = True
        else:
            self.de = False

        self.pof = config['of']['p']
        self.pue = config['ue']['p']

    @staticmethod
    def salt_pepper_noise(x, rate):
        x_ = x.clone()
        if x_[2].any() != 0:
            noise = torch.rand_like(x_)
            flipped = noise < rate
            salted = torch.rand_like(x_) > 0.5
            peppered = ~salted
            x_[flipped & salted] = 1
            x_[flipped & peppered] = 0
            return x_
        else:
            noise = torch.rand_like(x_[0])
            flipped = noise < rate
            salted = torch.rand_like(x_[0]) > 0.5
            peppered = ~salted
            x_[0, flipped & salted] = 1
            x_[1, flipped & peppered] = 1
            return x_

    @staticmethod
    def simulate_underexposure(event_frame, p_remove=0.2, p_noise=0.01):
        result = event_frame.clone()
        remove_mask = (result == 1) & (torch.rand_like(result, dtype=torch.float32) < p_remove)
        result[remove_mask] = 0

        noise_mask = torch.rand_like(result, dtype=torch.float32) < p_noise
        result[noise_mask] = 1

        return result

    def __call__(self, x, seed=None):
        scene = torch.zeros(1)
        if not self.de:
            return x, scene
        else:
            r = torch.rand(1)
            if seed is not None:
                torch.manual_seed(seed)
            if r < self.pof:
                x = EVDegrader.salt_pepper_noise(x, self.config['of']['rate'])
                scene[0] = 1
            elif r < self.pof + self.pue:
                x = EVDegrader.simulate_underexposure(x, self.config['ue']['p_remove'], self.config['ue']['p_noise'])
                scene[0] = 2
            return x, scene


class ComDegrader(object):
    def __init__(self, config, mode):
        self.config = config
        if mode == 'train' or mode == 'val':
            self.de = True
        else:
            self.de = False

        self.gaussianblur_aug = [
            config['gaussianblur']['p'],
            transforms.GaussianBlur(
                kernel_size=config['gaussianblur']['kernel_size'],
                sigma=config['gaussianblur']['sigma'],
            )
        ]

    @staticmethod
    def gaussian_noise(x, var_limit):
        if x[2].any() != 0:
            var = (torch.rand(1) * var_limit[1] - var_limit[0]) + var_limit[0]
            noise = torch.randn_like(x) * var
            return x + noise
        else:
            var = (torch.rand(1) * var_limit[1] - var_limit[0]) + var_limit[0]
            noise = torch.randn_like(x) * var
            x = torch.clip(x + noise, 0, 1.)
            x[2] = 0
            return x

    def __call__(self, x, seed=None):

        if not self.de:
            return x
        else:
            if seed is not None:
                torch.manual_seed(seed)
            if torch.rand(1) < self.gaussianblur_aug[0]:
                x = self.gaussianblur_aug[1](x)
            if torch.rand(1) < self.config['gauss']['p']:
                x = ComDegrader.gaussian_noise(x, var_limit=self.config['gauss']['var'])

            return x


class BaseRgbTransform():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        if std is None:
            std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        self.mean = np.array(mean).reshape((1, 1, 1, 3))
        self.std = np.array(std).reshape((1, 1, 1, 3))

    def __call__(self, x):
        return (x - self.mean) / self.std
