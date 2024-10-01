import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    '''
    P 行列から K と Rt を取り出す
    '''
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2] # scale で割ることで，内部パラメータを正規化
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K
    
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose() # world to image -> image to world
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def load_K_Rt_from_opensfm(filepath):
    '''
    内部パラメータと外部パラメータを OpenSfM の reconstruction.jsonから取り出す
    Input:
        filepath(str): ファイルパス
    Output:
        intrinsics: 内部パラメータ（360度画像の場合，内部パラメータは取り出さない TODO: 将来的には取り出せるようにする）
        pose: 外部パラメータ
    '''
    # TODO: 今後実装する
    intrinsics, pose = None, None

    return intrinsics, pose
class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        #
        # カメラに関する情報の取り出し
        #

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        # スケール行列だけは，3 次元復元を行うためには必要な作業かもしれない
        # TODO: ワールド行列とスケール行列の関係性を今一度理解したほうがいい
        # self.scale_mats_np = []
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        
        #
        # カメラと画像に関する初期化
        #

        self.is_erp_image = conf.get_bool('is_erp_image', default=False) # 360 度画像を入力とする場合に True とする
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W
        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx.cpu()][(pixels_y.cpu(), pixels_x.cpu())]    # batch_size, 3
        mask = self.masks[img_idx.cpu()][(pixels_y.cpu(), pixels_x.cpu())]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    # デバッグとして一時的にここに置く
    def calc_near_far_within_sphere(self, rays_o, rays_d, alpha=200, epsilon=0.01):
        '''
        注目領域（単位球）の中にカメラがある場合は，カメラ原点から微小に離れた点を near,
        単位球と交差する点までの距離を far としてレンダリング範囲を定義する
        '''
        inner_prod_od = torch.sum(rays_o*rays_d, dim=-1, keepdim=True)
        power_o = torch.sum(rays_o**2, dim=-1, keepdim=True)

        # t が大きい方（必然的に正の値）を単位球面への距離とする
        # t is the distance from the camera origin to the intersection point with the unit sphere
        far1 = -inner_prod_od - torch.sqrt(inner_prod_od**2 - power_o + 1)
        far2 = -inner_prod_od + torch.sqrt(inner_prod_od**2 - power_o + 1)
        far = torch.maximum(far1, far2)
        
        if torch.any(far < 0):
            raise ValueError("The value of far must be positive")
        
        far = far + alpha * epsilon
        # 試しに，固定値でやってみる
        # Try a constant value
        near = torch.full(far.shape, epsilon)
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

# Dataset クラスを継承して，360度画像用のクラスを作成
class ErpDataset(Dataset):
    def __init__(self, conf):
        super().__init__(conf)

        # self.is_erp_image = conf.get_bool('is_erp_image', default=False) # 360 度画像を入力とする場合に True とする

    def _calc_erp_viewdirs(self, pix, H, W, radius=1):
        '''
        p [H, W, 3] : [pixel_x, pixel_y, 1] for each pixel
        H           : Height of image
        W           : Width of image
        radius      : radius of sphere camera model (ideal radius is 1)
        '''
        # NeuS は，[right, down, front] で定義している気がする...
        # 画像平面に向かって z 軸が伸びるように，実装を変更した
        # lon = torch.pi * (2*(pix[..., 0] + 0.5)/W - 1) # [H, W] (-pi, pi)
        # lat = torch.pi * (0.5*H - (pix[..., 1] + 0.5)) / H # [H, W] (-pi/2, pi/2)
        lat = (pix[..., 1] + 0.5) * torch.pi / H - (torch.pi*0.5)
        lon = (2 * torch.pi * (pix[..., 0] + 0.5) / W) - torch.pi
        X = radius * torch.cos(lat) * torch.sin(lon)
        Y = radius * torch.sin(lat)
        Z = radius * torch.cos(lat) * torch.cos(lon)
        XYZ = torch.stack([X, Y, Z], 2).squeeze() # [H, W, 3]
        return XYZ
    
    def _calc_batch_erp_viewdirs(self, pix, H, W, radius=1):
        '''
        p [B, 3]    : a batch of [pixel_x, pixel_y, 1]
        H           : Height of image
        W           : Width of image
        radius      : radius of sphere camera model (ideal radius is 1)
        '''
        # lon = 2*torch.pi * (W - pix[:, 0]) / W # [B, 1]
        # lat = torch.pi * (0.5*H - pix[:, 1]) / H # [B, 1]
        lat = (pix[:, 1] + 0.5) * torch.pi / H - (torch.pi*0.5)
        lon = (2 * torch.pi * (pix[:, 0] + 0.5) / W) - torch.pi
        X = radius * torch.cos(lat) * torch.sin(lon)
        Y = radius * torch.sin(lat)
        Z = radius * torch.cos(lat) * torch.cos(lon)
        XYZ = torch.stack([X, Y, Z], 1).squeeze() # [B, 3]
        return XYZ

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        if self.is_erp_image:
            #
            # 360 度画像用の視線方向の定義
            #
            rays_v = self._calc_erp_viewdirs(p, self.H, self.W)
        else:
            p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze() # W, H, 3 # カメラ座標系における視線方向を定義
            rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])

        #
        # 360度画像用のランダム取り出し（TODO: 将来的には検討，実装する）
        #

        color = self.images[img_idx.cpu()][(pixels_y.cpu(), pixels_x.cpu())]    # batch_size, 3
        mask = self.masks[img_idx.cpu()][(pixels_y.cpu(), pixels_x.cpu())]      # batch_size, 3
        # colobr = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        # mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3 # バッチサイズ分の視線方向を初期化
        if self.is_erp_image:
            #
            # 360 度画像用の視線方向の定義
            #
            rays_v = self._calc_batch_erp_viewdirs(p, self.H, self.W)
        else:
            p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3 # カメラ座標系における視線方向を定義
            rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3 視線方向を表すベクトルを正規化（ノルムが 1 になるようにする）
        
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3 視線方向を世界座標系へ変換
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def calc_near_far_within_sphere(self, rays_o, rays_d, alpha=10, epsilon=0.001):
        '''
        注目領域（単位球）の中にカメラがある場合は，カメラ原点から微小に離れた点を near,
        単位球と交差する点までの距離を far としてレンダリング範囲を定義する
        '''
        inner_prod_od = torch.sum(rays_o*rays_d, dim=-1, keepdim=True)
        power_o = torch.sum(rays_o**2, dim=-1, keepdim=True)

        # t が大きい方（必然的に正の値）を単位球面への距離とする
        far1 = -inner_prod_od - torch.sqrt(inner_prod_od**2 - power_o + 1)
        far2 = -inner_prod_od + torch.sqrt(inner_prod_od**2 - power_o + 1)
        far = torch.maximum(far1, far2)
        
        if torch.any(far < 0):
            raise ValueError("The value of far must be positive")
        
        far = far + alpha * epsilon
        near = torch.full(far.shape, epsilon)
        return near, far
