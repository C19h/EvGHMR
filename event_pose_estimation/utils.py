import cv2
import numpy as np
import pandas as pd
import torch
from pytorch3d.renderer import (
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    BlendParams
)
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
import torchvision
import math
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml

# Color definitions
colors = {
    'pink': torch.tensor([0.7, 0.7, 0.9], dtype=torch.float32),
    'neutral': torch.tensor([0.9, 0.9, 0.8], dtype=torch.float32),
    'capsule': torch.tensor([0.7, 0.75, 0.5], dtype=torch.float32),
    'yellow': torch.tensor([0.5, 0.7, 0.75], dtype=torch.float32),
}


def rotate_y(points, angle):
    """Rotate vertices around the Y axis."""
    ry = torch.tensor([
        [np.cos(angle), 0., np.sin(angle)],
        [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ], dtype=torch.float32)
    return torch.matmul(points, ry.T)


def create_renderer(image_size=(256, 256)):
    """Create a Pytorch3D renderer."""
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        bin_size=0,
        faces_per_pixel=2
    )

    lights = PointLights(
        location=[[0, 5, -5]],  # Adjust light position
        diffuse_color=((0.5, 0.5, 0.5),),
        specular_color=((0.3, 0.3, 0.3),)
    )
    blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftPhongShader(lights=lights, blend_params=blend_params)
    )

    return renderer


def save_rendered_image(image_tensor, filename="output.png"):
    """
    Save a rendered image as a PNG or JPEG file.
    :param image_tensor: Rendered PyTorch tensor with shape (H, W, 3)
    :param filename: Output filename
    """
    # **Convert Tensor to NumPy array**
    image_np = image_tensor.squeeze(0).cpu().numpy()  # Remove batch dimension

    # **Check whether values are in [0,1], then convert to [0,255]**
    if image_np.max() <= 10.0:
        image_np = image_np / image_np.max()
        image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

    # **Convert channel format (PyTorch: RGB -> OpenCV: BGR)**
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # **Save image**
    cv2.imwrite(filename, image_np)
    print(f"渲染图像已保存为 {filename}")


def render_model(verts, faces, cam_param, cam_t, image_size=(256, 256), img=None, return_mesh=False):
    """
    Render a 3D model using pytorch3d.
    :param verts: Vertex coordinates [N, V, 3]
    :param faces: Triangle indices [N, F, 3]
    :param cam_param: Camera intrinsics (fx, fy, cx, cy)
    :param cam_t: Camera translation
    :param cam_rt: Camera rotation matrix
    :param image_size: Rendered image size
    :param img: Optional background image
    """

    # Parse camera parameters
    fx, fy, cx, cy = cam_param[:4]
    verts = torch.tensor(verts, dtype=torch.float32).unsqueeze(0)

    # Fix faces dtype
    faces = torch.tensor(faces.astype(np.int64), dtype=torch.int64).unsqueeze(0)

    # Camera intrinsic matrix
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).unsqueeze(0)

    # Rotation & translation
    R = torch.eye(3, dtype=torch.float32).unsqueeze(0)  # Rotation
    R = R.clone().detach()
    T = torch.tensor(cam_t, dtype=torch.float32).unsqueeze(0)  # Translation

    # Build camera
    cameras = cameras_from_opencv_projection(
        R=R, tvec=T, camera_matrix=K, image_size=torch.tensor(image_size).expand(1, 2))

    verts_rgb = colors['pink'].expand(verts.shape[0], verts.shape[1], 3)
    textures = TexturesVertex(verts_rgb)

    # Construct mesh object
    mesh = Meshes(verts=verts, faces=faces, textures=textures)

    # Create renderer
    renderer = create_renderer(image_size=image_size)

    # Render
    images = renderer(mesh, cameras=cameras)
    # frags = renderer.rasterizer(mesh, cameras=cameras)

    # Blend with background image if provided
    if img is not None:
        img = torch.tensor(img, dtype=torch.float32) / 255.0  # Normalize
        img = img.unsqueeze(0)  # Adjust dimensions
        mask = images[..., 3:]  # Alpha channel
        images = images[..., :3] * mask + img * (1 - mask)

    image_np = images[..., :3].squeeze(0).cpu().numpy()  # Remove batch dimension

    # **Check whether values are in [0,1], then convert to [0,255]**
    if image_np.max() <= 10.0:
        image_np = image_np / image_np.max()
        image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

    # **Convert channel format (PyTorch: RGB -> OpenCV: BGR)**
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    if return_mesh:
        return image_np, mesh
    return image_np  # Return RGB channels only


def get_single_image_crop(image, bbox, scale=1.2, crop_size=224):
    if isinstance(image, str):
        if os.path.isfile(image):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        else:
            print(image)
            raise BaseException(image, 'is not a valid file!')
    elif isinstance(image, torch.Tensor):
        image = image.numpy()
    elif not isinstance(image, np.ndarray):
        raise ('Unknown type for object', type(image))

    crop_image, trans = generate_patch_image_cv(
        cvimg=image.copy(),
        c_x=bbox[0],
        c_y=bbox[1],
        bb_width=bbox[2],
        bb_height=bbox[3],
        patch_width=crop_size,
        patch_height=crop_size,
        do_flip=False,
        scale=scale,
        rot=0,
    )

    return crop_image


def generate_patch_image_cv(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip, scale, rot):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, inv=False)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return img_patch, trans


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y  # np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def adjust_2d_joints_and_camera(bbox, bbox_scale, joints_2d, camera_intrinsics):
    center_x, center_y, bbox_size, _ = bbox

    joints_2d[:, 0] -= (center_x - bbox_size / 2)
    joints_2d[:, 1] -= (center_y - bbox_size / 2)

    adjusted_joints_2d = joints_2d * bbox_scale

    fx, fy, cx, cy = camera_intrinsics

    fx_scaled = fx * bbox_scale
    fy_scaled = fy * bbox_scale

    cx_scaled = (cx - (center_x - bbox_size / 2)) * bbox_scale
    cy_scaled = (cy - (center_y - bbox_size / 2)) * bbox_scale

    adjusted_camera_intrinsics = np.array([fx_scaled, fy_scaled, cx_scaled, cy_scaled])

    return adjusted_joints_2d, adjusted_camera_intrinsics
class ConfigParser:
    # YMAL parser for configs
    def __init__(self, config):
        self._config = {}
        self.reset_config()
        self.parse_config(config)

    def parse_config(self, file):
        with open(file, 'r') as fid:
            yaml_config = yaml.safe_load(fid)
        self.parse_dict(yaml_config)

    def parse_dict(self, input_dict, parent=None):
        if parent is None:
            parent = self._config
        for key, val in input_dict.items():
            if isinstance(val, dict):
                if key not in parent.keys():
                    parent[key] = {}
                self.parse_dict(val, parent[key])
            else:
                parent[key] = val

    def reset_config(self):
        self._config['data'] = {}
        self._config['exper'] = {}
        self._config['model'] = {}
        self._config['eval'] = {}
        self._config['utils'] = {}

    def save_config(self, path_dir, file_name):
        with open(os.path.join(path_dir, file_name), 'w') as fid:
            yaml.dump(self._config, fid)

    @staticmethod
    def save_config_dict(config_dict, path_dir, file_name):
        with open(os.path.join(path_dir, file_name), 'w') as fid:
            yaml.dump(config_dict, fid)

    def update(self, config):
        self.reset_config()
        self.parse_config(config)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    def merge_configs(self, path_models):
        # parse training config
        with open(path_models, 'r') as fid:
            config = yaml.safe_load(fid)
        # overwrite with config settings
        self.parse_dict(config, self._config)
        return config


def compute_optical_flow(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    return flow


def warp_image_with_flow(image, flow, t):
    h, w = image.shape[:2]

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    map_x = (grid_x + t * flow[..., 0]).astype(np.float32)
    map_y = (grid_y + t * flow[..., 1]).astype(np.float32)
    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def interpolate_frames(frame1, frame2, num_interp=5):
    flow = compute_optical_flow(frame1, frame2)

    t_values = np.linspace(0, 1, num_interp)
    warped_frames = []
    for t in t_values:
        warped = warp_image_with_flow(frame1, flow, t)
        warped_frames.append(warped.astype(np.float32))

    avg_frame = np.mean(warped_frames, axis=0)
    avg_frame = np.clip(avg_frame, 0, 255).astype(np.uint8)
    return avg_frame


def simulate_motion_blur_from_frames(frames, num_interp=5):
    blurred_frames = []

    for i in range(len(frames) - 1):
        mb_frame = interpolate_frames(frames[i], frames[i + 1], num_interp)
        blurred_frames.append(mb_frame.astype(np.float32))

    if blurred_frames:
        final_blur = np.mean(blurred_frames, axis=0)
        final_blur = np.clip(final_blur, 0, 255).astype(np.uint8)
    else:
        final_blur = frames[0]
    return final_blur


def get_eventsteam(file):
    events = pd.read_csv(file, header=None, dtype=np.float32,
                         names=['v', 'u', 'in_pixel_time', 'off_pixel_time', 'polarity'])
    events.dropna(inplace=True)
    events['in_pixel_time'] = events['in_pixel_time'].astype(np.int32)
    events['u'] = events['u'].astype(np.int32)
    events['v'] = events['v'].astype(np.int32)
    events = events[(events['v'] <= 1279) &
                    (events['u'] <= 799)]
    events = events[['v', 'u', 'in_pixel_time']].values
    sorted_indices = np.argsort(events[:, 2])
    events = events[sorted_indices]
    return events


def generate_events_frame(events, num_partitions=4, h=1280, w=800):
    # events [N, 3], v, u, in_pixel_time
    start_time = events[0, 2]
    end_time = events[-1, 2]
    time_window = (end_time - start_time) / num_partitions
    events_frames = []
    for i in range(num_partitions):
        events_frame = np.zeros([h, w])
        idx = (events[:, 2] >= (start_time + i * time_window)) & (events[:, 2] < (start_time + (i + 1) * time_window))
        v = h - 1 - events[idx, 0]
        u = w - 1 - events[idx, 1]
        events_frame[v, u] = 1
        events_frames.append(events_frame)
    events_frames = np.stack(events_frames, axis=2)
    return events_frames


def events2Tore3C(events, k, frameSize):
    x = events[:,0]
    y = events[:,1]
    ts = events[:,2]
    ts = ts - np.min(ts)
    max_time = np.max(ts)
    sampleTimes = [max_time]
    toreFeature = np.inf * np.ones((frameSize[0], frameSize[1], k))
    Xtore = np.zeros((frameSize[0], frameSize[1], k, len(sampleTimes)), dtype=np.single)
    priorSampleTime = -np.inf

    for sampleLoop, currentSampleTime in enumerate(sampleTimes):
        addEventIdx = (ts >= priorSampleTime) & (ts < currentSampleTime)

        newTore = np.full((frameSize[0], frameSize[1], k), np.inf)
        for i, j, t in zip(x[addEventIdx], y[addEventIdx], ts[addEventIdx]):
            i = frameSize[0]-1-i
            j = frameSize[1]-1-j
            newTore[i, j] = np.sort(np.partition(np.append(newTore[i, j], currentSampleTime - t), k)[:k])

        toreFeature += (currentSampleTime - priorSampleTime)
        toreFeature = np.sort(np.concatenate((toreFeature, newTore), axis=2), axis=2)[:, :, :k]

        Xtore[:, :, :, sampleLoop] = toreFeature.astype(np.single)

        priorSampleTime = currentSampleTime

    # Scale the Tore surface
    minTime = 150
    maxTime = 5e6

    Xtore[np.isnan(Xtore)] = maxTime
    Xtore[Xtore > maxTime] = maxTime

    Xtore = np.log(Xtore + 1)
    Xtore = Xtore - np.log(minTime + 1)
    Xtore[Xtore < 0] = 0

    return Xtore

def simulate_low_light_events(events, drop_prob=0.8, noise_rate=0.01, time_range=None):
    """
    Simulate event streams under low-light conditions:
      - Randomly drop a portion of real events (drop_prob)
      - Add a small amount of random noise events (noise_rate)

    Args:
      events: Original event stream of shape (N, 3), each event is [x, y, t]
      drop_prob: Probability of dropping real events (e.g., 0.5 means dropping 50%)
      noise_rate: Ratio of additional random noise events relative to original event count
      time_range: If None, use min/max t from events to generate noise timestamps;
                  otherwise pass (t_min, t_max)
    Returns:
      Simulated low-light event stream of shape (M, 3)
    """
    # Subsample real events
    mask = np.random.rand(len(events)) > drop_prob
    events_low = events[mask]

    # Generate noise events
    num_noise = int(len(events) * noise_rate)
    if time_range is None:
        t_min, t_max = events[:, 2].min(), events[:, 2].max()
    else:
        t_min, t_max = time_range
    noise_x = np.random.randint(0, events[:, 0].max() + 1, size=num_noise)
    noise_y = np.random.randint(0, events[:, 1].max() + 1, size=num_noise)
    noise_t = np.random.uniform(t_min, t_max, size=num_noise)
    noise_events = np.stack((noise_x, noise_y, noise_t), axis=1)

    # Merge real/noise events and sort by time
    simulated_events = np.concatenate((events_low, noise_events), axis=0)
    simulated_events = simulated_events[np.argsort(simulated_events[:, 2])]

    return simulated_events


def simulate_overexposed_events(events, extra_rate=0.1, time_range=None):
    events_over = events.copy()

    # Generate additional background noise events
    num_extra = int(len(events) * extra_rate)
    if time_range is None:
        t_min, t_max = events[:, 2].min(), events[:, 2].max()
    else:
        t_min, t_max = time_range
    # Assume the whole image is background and sample noise from x/y ranges
    noise_x = np.random.randint(0, events[:, 0].max() + 1, size=num_extra)
    noise_y = np.random.randint(0, events[:, 1].max() + 1, size=num_extra)
    noise_t = np.random.uniform(t_min, t_max, size=num_extra)
    extra_events = np.stack((noise_x, noise_y, noise_t), axis=1)

    # Merge events and sort by time
    simulated_events = np.concatenate((events_over, extra_events), axis=0)
    simulated_events = simulated_events[np.argsort(simulated_events[:, 2])]

    return simulated_events

def get_es_file(old_path):
    subject_folder = os.path.basename(os.path.dirname(old_path))
    filename = os.path.basename(old_path)
    filename_without_ext, _ = os.path.splitext(filename)
    new_filename = filename_without_ext + '.csv'
    new_path = os.path.join(
        '../data_event',
        'data_event_raw',
        subject_folder,
        'event_camera',
        'events',
        new_filename
    )
    return new_path


class SAFusionBlock(nn.Module):
    def __init__(self, inChannels, outChannels, radius=3):
        super(SAFusionBlock, self).__init__()
        self.radius = radius
        self.complement = nn.Sequential(*[
            nn.Conv2d((2 * radius + 1) ** 2 + inChannels, inChannels, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1, stride=1)])

    def get_costvolume(self, ipt1, ipt2):
        b, c, h, w = ipt1.shape
        # cost volume
        Fi_unfold = F.unfold(ipt1, kernel_size=self.radius * 2 + 1, padding=self.radius)
        Fi_unfold = Fi_unfold.view(b, c, -1, h, w)
        Fi_unfold = Fi_unfold.permute(0, 3, 4, 2, 1)
        ipt2_ = ipt2.permute(0, 2, 3, 1)
        cost_volume = torch.einsum('bhwc,bhwdc->bhwd', ipt2_, Fi_unfold)
        cost_volume = cost_volume.permute(0, 3, 1, 2)
        cost_volume /= torch.sqrt(torch.tensor(c).float())
        return cost_volume

    def forward(self, Fi, Fe):
        feat_rgb, feat_ev = Fi, Fe
        i_e = self.get_costvolume(feat_rgb, feat_ev)
        e_i = self.get_costvolume(feat_ev, feat_rgb)

        f_i_e = self.complement(torch.cat((i_e, feat_ev), 1))
        f_e_i = self.complement(torch.cat((e_i, feat_rgb), 1))

        # channel spatial attention
        # aligned_ev, aligned_rgb = self.channel_spatial_attention(f_i_e, f_e_i)

        # fusion
        # feat_cat = torch.cat((aligned_rgb, aligned_ev), dim=1)
        # Ff = self.fusion(feat_cat)

        return f_i_e, f_e_i


class PreImg(nn.Module):
    def __init__(self, name: str,
                 dilation: bool,
                 in_c=3,
                 pretrained=False,
                 ):
        super(PreImg, self).__init__()
        norm_layer = nn.BatchNorm2d
        resnet = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation], norm_layer=norm_layer, pretrained=pretrained)
        resnet.conv1 = torch.nn.Conv2d(in_c, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

    def forward(self, x):
        c1 = self.layer0(x)
        return c1

class PreEv(nn.Module):
    def __init__(self, name: str,
                 dilation: bool,
                 in_c=8):
        super(PreEv, self).__init__()
        norm_layer = nn.BatchNorm2d
        resnet = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation], norm_layer=norm_layer)
        resnet.conv1 = torch.nn.Conv2d(in_c, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

    def forward(self, x):
        c1 = self.layer0(x)
        return c1


class PostBackbone(nn.Module):
    def __init__(self, name: str,
                 dilation: bool):
        super(PostBackbone, self).__init__()
        norm_layer = nn.BatchNorm2d
        resnet = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation], norm_layer=norm_layer)
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)
        self.layer5 = nn.Sequential(resnet.avgpool)

    def forward(self, x):
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        c6 = self.layer5(c5)
        return c6

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class AttentionFusion(nn.Module):
    def __init__(self, in_channels, squeeze_ratio, feat_len):
        super(AttentionFusion, self).__init__()
        hidden_dim = int(in_channels / squeeze_ratio)
        self.feat_len = feat_len
        self.conv = nn.Sequential(
                conv1x1(in_channels * feat_len, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                conv3x3(hidden_dim, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                conv1x1(hidden_dim, in_channels * feat_len),
            )

    def forward(self, feat_list):
        '''
            sil_feat: [ns, c, h, w]
            map_feat: [ns, c, h, w]
            ...
        '''
        feats = torch.cat(feat_list, dim=1)
        score = self.conv(feats)  # [ns, 2 * c, h, w]
        score = rearrange(score, 'ns (c d) h w -> ns c d h w', d=self.feat_len)
        score = F.softmax(score, dim=2)
        retun = feat_list[0] * score[:, :, 0]
        for i in range(1, self.feat_len):
            retun += feat_list[i] * score[:, :, i]
        return retun, score
