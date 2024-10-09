import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def visualize_cameras(camera_params):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # カメラの可視化に使う配列の処理
    N_images = camera_params.shape[0]
    
    for i in range(N_images):
        # 3x3の回転行列Rと並進ベクトルTを取得
        R = camera_params[i, :, :3]
        T = camera_params[i, :, 3]
        h = camera_params[i, 0, 4]  # 画像の高さ
        w = camera_params[i, 1, 4]  # 画像の幅
        f = camera_params[i, 2, 4]  # 焦点距離
        
        # カメラ座標系での四角錐の点を定義
        # カメラの中心(原点)は(0,0,0)
        # 底面の四角形の頂点をカメラ座標系に定義
        # z軸が画像の奥行き方向なので、カメラの画像平面はz = fの位置
        camera_corners = np.array([
            [-w / 2, -h / 2, f],  # 左下
            [ w / 2, -h / 2, f],  # 右下
            [ w / 2,  h / 2, f],  # 右上
            [-w / 2,  h / 2, f],  # 左上
        ])
        
        # カメラの頂点（ピラミッドの頂点）は原点(0, 0, 0)にあるので省略可能
        apex = np.array([0, 0, 0])

        # カメラ座標系からワールド座標系への変換
        camera_corners_world = (R @ camera_corners.T).T + T
        apex_world = T

        # カメラの四角錐の頂点を可視化
        for corner in camera_corners_world:
            ax.plot([apex_world[0], corner[0]], [apex_world[1], corner[1]], [apex_world[2], corner[2]], 'k-')

        # 底面（四角形）を描画
        verts = [camera_corners_world]
        ax.add_collection3d(Poly3DCollection(verts, color='cyan', linewidths=1, edgecolors='r', alpha=0.1))
    
    # 軸のラベル
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 等間隔に表示
    ax.set_box_aspect([1, 1, 1])
    plt.show()

# サンプルデータ (N_images=2, 3x5の外部パラメータ)
camera_params = np.array([
    [[1, 0, 0, 1, 500], [0, 1, 0, 2, 400], [0, 0, 1, 3, 800]],
    [[0.866, -0.5, 0, 4, 500], [0.5, 0.866, 0, 5, 400], [0, 0, 1, 6, 800]]
])

visualize_cameras(camera_params)
