from torchvision.models import resnet50
from DINOv2 import vit_small
from SMPL import SMPL
from geometry import projection_torch, rot6d_to_rotmat
from utils import *
import matplotlib.pyplot as plt
import PIL.Image as Image
class MyTemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True
    ):
        super(MyTemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers,
            batch_first=True
        )

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size * 2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual

    def forward(self, x):
        # x: [B, T, F]
        # hidden_feats: [B, 2048]
        B, T, fe = x.shape
        y, _ = self.gru(x)  # y: [B, T, F]
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.reshape(-1, y.size(-1)))
            y = y.view(B, T, fe)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        return y

class MyRegressor(nn.Module):
    def __init__(self, in_c, pose_dim=24 * 6):
        super(MyRegressor, self).__init__()
        # self.fc1 = nn.Linear(2048, 1024)
        # self.decpose = nn.Linear(1024, pose_dim)
        # self.dectrans = nn.Linear(1024, 3)
        self.decpose = nn.Linear(in_c, pose_dim)
        self.dectrans = nn.Linear(in_c, 3)
        self.decshape = nn.Linear(in_c, 10)

    def forward(self, x):
        # x: [B, T, 2048]
        B, T, F = x.size()
        x = x.reshape(-1, F)
        # x = func.dropout(self.fc1(x))
        x1 = self.decpose(x)  # [B*T, pose_dim]
        x2 = self.dectrans(x)  # [B*T, 3]
        x3 = self.decshape(x)  # [B*T, 10]
        pose = x1.view(B, T, x1.size(-1))
        tran = x2.view(B, T, x2.size(-1))
        shape = x3.view(B, T, x3.size(-1))
        return pose, tran, shape

class MyVIBERegressor(nn.Module):
    def __init__(self, pose_dim=24 * 6, smpl_mean_params='../smpl_model/smpl_mean_params.npz'):
        super(MyVIBERegressor, self).__init__()
        self.fc1 = nn.Linear(2048 + pose_dim + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, pose_dim)
        self.dectrans = nn.Linear(1024, 3)
        self.decshape = nn.Linear(1024, 10)

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose']).float().unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape']).float().unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).float().unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_trans', init_cam)
        self.register_buffer('init_shape', init_shape)

    def forward(self, x, n_iter=3):
        # x: [B, T, 2048]
        B, T, F = x.size()
        x = x.view(-1, F)

        init_pose = self.init_pose.expand(B * T, -1)
        init_trans = self.init_trans.expand(B * T, -1)
        init_shape = self.init_shape.expand(B * T, -1)

        pred_pose = init_pose
        pred_trans = init_trans
        pred_shape = init_shape
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_trans, pred_shape], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose  # [B*T, pose_dim]
            pred_trans = self.dectrans(xc) + pred_trans  # [B*T, 3]
            pred_shape = self.decshape(xc) + pred_shape
        pose = pred_pose.view(B, T, pred_pose.size(-1))
        trans = pred_trans.view(B, T, pred_trans.size(-1))
        shape = pred_shape.view(B, T, pred_shape.size(-1))
        return pose, trans, shape

class EventTrackNetVanilla(nn.Module):
    def __init__(
            self,
            events_input_channel=8,
            smpl_dir='../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl',
            n_layers=1,
            hidden_size=2048,
            bidirectional=False,
            add_linear=False,
            use_residual=True,
            pose_dim=24 * 6,
            use_flow=True,
            vibe_regressor=False,
            smpl_mean_params='../smpl_model/smpl_mean_params.npz',
            logger = None
    ):
        super(EventTrackNetVanilla, self).__init__()
        self.img_encoder = self.resnet50_encoder(events_input_channel + 2 * use_flow)  # 2048
        self.use_flow = use_flow
        self.tmp_encoder = MyTemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )

        self.vibe_regressor = vibe_regressor
        if self.vibe_regressor:
            self.regressor = MyVIBERegressor(pose_dim, smpl_mean_params)
        else:
            self.regressor = MyRegressor(2048, pose_dim)
        self.smpl = SMPL(smpl_dir)
        self.init_parameters()
        n_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info('All Trainable Count: {:.5f}M'.format(n_trainable_params / 1e6))

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def resnet50_encoder(self, input_channel):
        model = resnet50(pretrained=False)
        model.conv1 = torch.nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Sequential()
        return model

    def forward(self, data):
        # events, pred_flow: [B, T, C, H, W]
        # init_shape: [B, 1, 85]
        # hidden_feats: [B, 2048]
        # print(torch.isnan(events).any())
        events = data['events']
        cam_intr = data['cam_intr']
        B, T, C, H, W = events.size()
        if self.use_flow:
            pred_flow = data['flows']
            # [B*T, 8+2/3, H, W]
            x = torch.cat([events, pred_flow], dim=2).view(-1, events.size(2) + pred_flow.size(2), H, W)
        else:
            x = events.view(-1, events.size(2), H, W)
        x = self.img_encoder(x).view(B, T, 2048)
        # print(torch.isnan(x).any())
        x = self.tmp_encoder(x).contiguous()  # [B, T, 2048]

        pose, trans, shape = self.regressor(x)  # pose [B, T, 24*6], tran [B, T, 3]
        # print(torch.isnan(pose).any())
        trans = trans.unsqueeze(dim=2)  # [B, T, 1, 3]
        pred_rotmats = rot6d_to_rotmat(pose.contiguous()).view(B, T, 24, 3, 3).contiguous()

        verts, joints3d, _ = self.smpl(beta=shape.view(-1, 10),
                                       theta=None,
                                       get_skin=True,
                                       rotmats=pred_rotmats.view(-1, 24, 3, 3))
        verts = verts.view(B, T, verts.size(1), verts.size(2)).contiguous() + trans.detach()

        results = {}
        results['shape'] = shape
        results['pred_rotmats'] = pred_rotmats  # [B, T, 24, 3, 3]
        results['tran'] = trans  # [B, T, 1, 3]
        results['verts'] = verts  # [B, T, 6890, 3]
        results['joints3d'] = joints3d.view(B, T, joints3d.size(1), joints3d.size(2)) + trans.detach()  # [B, T, 24, 3]
        if cam_intr is not None:
            results['joints2d'] = projection_torch(results['joints3d'], cam_intr, H, W)  # [B, T, 24, 2]
            results['cam_intr'] = cam_intr
        return results

class ComplNet_(nn.Module):
    def __init__(
            self,
            events_input_channel=8,
            smpl_dir='../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl',
            pose_dim=24 * 6,
            resnet='resnet50',
            mid4=True,
            logger=None,
    ):
        super(ComplNet_, self).__init__()
        self.mid4 = mid4
        self.dino_dim = 384
        if mid4:
            self.gconv = nn.Conv2d(self.dino_dim*4, 16, kernel_size=1, stride=1, padding=0)
        else:
            self.gconv = nn.Conv2d(self.dino_dim, 16, kernel_size=1, stride=1, padding=0)
        self.econv = nn.Conv2d(events_input_channel, 16, kernel_size=1, stride=1, padding=0)
        self.align_module = SAFusionBlock(64, 64)
        self.fusion = AttentionFusion(in_channels=64, squeeze_ratio=16, feat_len=2)
        self.preev = PreEv(name=resnet, dilation=False, in_c=16)
        self.preimg = PreImg(name=resnet, dilation=False, in_c=16)
        self.post_backbone = PostBackbone(name=resnet, dilation=False)
        self.tmp_encoder = MyTemporalEncoder(
            n_layers=1,
            hidden_size=2048,
            bidirectional=False,
            add_linear=False,
            use_residual=True,
        )
        self.regressor = MyRegressor(2048, pose_dim)
        self.smpl = SMPL(smpl_dir)
        self.init_parameters()
        ################Dinov2#######################################
        self.dino = vit_small(logger=logger)
        pretrain_dict = torch.load('./pretrained/dinov2_vits14_pretrain.pth')
        self.dino.load_state_dict(pretrain_dict, strict=True)
        self.dino.eval()
        self.dino.requires_grad_(False)
        n_parameters = sum(p.numel() for p in self.dino.parameters())
        logger.info('DINOv2 Count: {:.5f}M'.format(n_parameters / 1e6))
        ################Dinov2#######################################
        n_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info('All Trainable Count: {:.5f}M'.format(n_trainable_params / 1e6))

    def preprocess(self, x, image_size, mode='bilinear'):
        # shape: [nxs,c,h,w] / [nxs,c,224,112]
        return F.interpolate(x, (image_size, image_size), mode=mode, align_corners=False)

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, data):
        # events, pred_flow: [B, T, C, H, W]
        events = data['events']
        cam_intr = data['cam_intr']
        grays = data['grays'].permute(0, 1, 4, 2, 3)
        B, T, C, H, W = events.size()

        ef = events.view(-1, events.size(2), H, W)
        grays = grays.view(-1, grays.size(2), H, W)
        s_f = self.dino(grays, is_training=True)
        if not self.mid4:
            s_f_last = s_f["x_norm_patchtokens"].contiguous()
        else:
            s_f_last = s_f["x_norm_patchtokens_mid4"].contiguous()
        grays_f = rearrange(s_f_last.view(B, T, H // 14, W // 14, -1),
                            'n s h w c -> (n s) c h w').contiguous()
        # outs_c1 = self.preprocess(c1, H)
        # vis_c_f_last1 = pca(c1)
        # ind = 5
        # vis(vis_c_f_last1.transpose(0, 2, 3, 1), ind, f'./sf_{ind}.jpg')
        # vis_gray(vis_c_f_last1[10].transpose(1, 2, 0), './sf.png')
        # vis_gray(grays[10].cpu().detach().permute(1, 2, 0), './gray.png')
        # vis_gray(x[10,:3,...].cpu().detach().permute(1, 2, 0), './ef.png')

        ef_f = self.econv(ef)
        grays_f = self.gconv(grays_f)

        # ef_f = self.preprocess(ef_f, 128)
        grays_f = self.preprocess(grays_f, 128)

        g_f1 = self.preimg(grays_f)
        ef_f1 = self.preev(ef_f)
        app, ac = self.align_module(g_f1, ef_f1)
        fused_feat,score = self.fusion([app, ac])
        bbout = self.post_backbone(fused_feat)
        final_f = self.tmp_encoder(bbout.view(B, T, 2048)).contiguous()

        pose, trans, shape = self.regressor(final_f)  # pose [B, T, 24*6], tran [B, T, 3]
        # print(torch.isnan(pose).any())
        trans = trans.unsqueeze(dim=2)  # [B, T, 1, 3]
        pred_rotmats = rot6d_to_rotmat(pose.contiguous()).view(B, T, 24, 3, 3).contiguous()

        verts, joints3d, _ = self.smpl(beta=shape.view(-1, 10),
                                       theta=None,
                                       get_skin=True,
                                       rotmats=pred_rotmats.view(-1, 24, 3, 3))
        verts = verts.view(B, T, verts.size(1), verts.size(2)).contiguous() + trans.detach()

        results = {}
        results['shape'] = shape
        results['score'] = score
        results['pred_rotmats'] = pred_rotmats  # [B, T, 24, 3, 3]
        results['tran'] = trans  # [B, T, 1, 3]
        results['verts'] = verts  # [B, T, 6890, 3]
        results['joints3d'] = joints3d.view(B, T, joints3d.size(1), joints3d.size(2)) + trans.detach()  # [B, T, 24, 3]
        if cam_intr is not None:
            results['joints2d'] = projection_torch(results['joints3d'], cam_intr, H, W)  # [B, T, 24, 2]
            results['cam_intr'] = cam_intr
        return results

class ComplNetwoDino(nn.Module):
    def __init__(
            self,
            events_input_channel=8,
            smpl_dir='../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl',
            pose_dim=24 * 6,
            resnet='resnet50',
            logger=None,
    ):
        super(ComplNetwoDino, self).__init__()
        self.preev = PreEv(name=resnet, dilation=False, in_c=events_input_channel)
        self.post_backbone = PostBackbone(name=resnet, dilation=False)
        self.align_module = SAFusionBlock(64, 64)
        self.fusion = AttentionFusion(in_channels=64, squeeze_ratio=16, feat_len=2)
        # self.tematt = TemporalAttentionWithPatch(embed_dim=2048, num_heads=8, dropout=0.1, patch_size=3, num_layers=1)
        self.tmp_encoder = MyTemporalEncoder(
            n_layers=1,
            hidden_size=2048,
            bidirectional=False,
            add_linear=False,
            use_residual=True,
        )
        self.regressor = MyRegressor(2048, pose_dim)
        self.smpl = SMPL(smpl_dir)
        self.init_parameters()
        self.preimg = PreImg(name=resnet, dilation=False, pretrained=True)
        n_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info('All Trainable Count: {:.5f}M'.format(n_trainable_params / 1e6))

    def preprocess(self, x, image_size, mode='bilinear'):
        # shape: [nxs,c,h,w] / [nxs,c,224,112]
        return F.interpolate(x, (image_size, image_size), mode=mode, align_corners=False)

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, data):
        # events, pred_flow: [B, T, C, H, W]
        events = data['events']
        cam_intr = data['cam_intr']
        grays = data['grays'].permute(0, 1, 4, 2, 3)
        B, T, C, H, W = events.size()

        ef = events.view(-1, events.size(2), H, W)
        grays = grays.view(-1, grays.size(2), H, W)
        g_f1 = self.preimg(grays)
        ef_f1 = self.preev(ef)
        app, ac = self.align_module(g_f1, ef_f1)
        fused_feat = self.fusion([app, ac])
        bbout = self.post_backbone(fused_feat)
        final_f = self.tmp_encoder(bbout.view(B, T, 2048)).contiguous()
        # fianl_f = self.tematt(bbout)
        pose, trans, shape = self.regressor(final_f)  # pose [B, T, 24*6], tran [B, T, 3]
        # print(torch.isnan(pose).any())
        trans = trans.unsqueeze(dim=2)  # [B, T, 1, 3]
        pred_rotmats = rot6d_to_rotmat(pose.contiguous()).view(B, T, 24, 3, 3).contiguous()

        verts, joints3d, _ = self.smpl(beta=shape.view(-1, 10),
                                       theta=None,
                                       get_skin=True,
                                       rotmats=pred_rotmats.view(-1, 24, 3, 3))
        verts = verts.view(B, T, verts.size(1), verts.size(2)).contiguous() + trans.detach()

        results = {}
        results['shape'] = shape
        results['pred_rotmats'] = pred_rotmats  # [B, T, 24, 3, 3]
        results['tran'] = trans  # [B, T, 1, 3]
        results['verts'] = verts  # [B, T, 6890, 3]
        results['joints3d'] = joints3d.view(B, T, joints3d.size(1), joints3d.size(2)) + trans.detach()  # [B, T, 24, 3]
        if cam_intr is not None:
            results['joints2d'] = projection_torch(results['joints3d'], cam_intr, H, W)  # [B, T, 24, 2]
            results['cam_intr'] = cam_intr
        return results

def vis(total_frames, ind, pth):
    if len(total_frames.shape) == 3:
        norm_X = normalize_to_image(total_frames[ind])
    else:
        if total_frames.shape[-1] >3:
            norm_X = normalize_to_image(total_frames[ind][..., -3:])
        else:
            norm_X = normalize_to_image(total_frames[ind])
    img = Image.fromarray(norm_X)
    img.save(pth)

def normalize_to_image(array):
    array = np.squeeze(array)

    min_val = np.min(array)
    max_val = np.max(array)
    normalized = (array - min_val) / (max_val - min_val) * 255

    image = normalized.astype(np.uint8)

    return image

def vis_gray(image, path):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    data_min, data_max = image.min(), image.max()
    image = (image - data_min) / (data_max - data_min)
    image = (image * 255).astype(np.uint8)
    plt.imshow(image, cmap='viridis')  # Use a grayscale-like colormap
    plt.colorbar()  # Add a color bar to show the value range
    plt.title('Grayscale Image')
    plt.savefig(path)
    plt.close()


def pca(f):
    f4_numpy = f.detach().cpu().numpy()  # Convert tensor to a NumPy array
    n, c, h, w = f4_numpy.shape  # n: batch size, c: channels, h/w: feature map height/width

    # Flatten feature maps in the batch to a 2D array with shape (n * h * w, c)
    reshaped = f4_numpy.reshape(n, c, h * w).transpose(0, 2, 1).reshape(-1, c)  # (n * h * w, c)

    # Run PCA to reduce from c dimensions to 3
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(reshaped)  # (n * h * w, 3)

    # Reshape back to (n, h, w, 3)
    pca_images = pca_result.reshape(n, h, w, 3).transpose(0, 3, 1, 2)

    # Normalize PCA results to [0, 1]
    pca_images = (pca_images - pca_images.min()) / (pca_images.max() - pca_images.min())

    # Return PCA results for the whole batch
    return pca_images
