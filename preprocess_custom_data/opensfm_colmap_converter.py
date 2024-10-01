# Conversion from OpenSfM to COLMAP format
# Implemented by Shintaro Ito referrencing the COLMAP official document

import os
import json
import numpy as np
import math

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]]) 

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def angle_axis_to_quaternion(angle_axis: np.ndarray):
    angle = np.linalg.norm(angle_axis) # 回転角度を求める
    x = angle_axis[0] / angle # 回転軸の単位ベクトル
    y = angle_axis[1] / angle
    z = angle_axis[2] / angle

    qw = math.cos(angle / 2.0)
    qx = x * math.sqrt(1 - qw * qw)
    qy = y * math.sqrt(1 - qw * qw)
    qz = z * math.sqrt(1 - qw * qw)

    return np.array([qw, qx, qy, qz])

def ConvertFormat(dir):

    reconstruction_file = os.path.join(dir, 'reconstruction.json')

    cameras = {}
    images = {}
    points3ds = {}
    image_id = 1
    camera_id = 1
    point_id = 1
    with open(reconstruction_file) as f:
        reconstructions = json.load(f)

        for reconstruction in reconstructions: # 通常，reconstructions の中身は１つ
            for camera in reconstruction["cameras"]:
                
                camera_info = reconstruction["cameras"][camera]
                
                # intrinsics
                if camera_info['projection_type'] == "spherical":
                    model = "SPHERICAL" # カメラモデル model
                    width = camera_info["width"] # 幅 W
                    height = camera_info["height"] # 高さ H
                    params = []
                elif camera_info['projection_type'] == "perspective":
                    model = "SIMPLE_PINHOLE" # カメラモデル model
                    width = camera_info["width"] # 幅 W
                    height = camera_info["height"] # 高さ H
                    f = camera_info["focal"] * width # 焦点距離 focal
                    k1 = camera_info["k1"]
                    k2 = camera_info["k2"]
                    params = [f, k1, k2]

                # cameras[camera_name] = {camera_id, model, width, height, params} # cameras.txt
                cameras[camera] = {
                    "CAMERA_ID": camera_id,
                    "MODEL": model,
                    "WIDTH": width,
                    "HEIGHT": height,
                    "PARAMS": params
                }
                camera_id += 1
            
            for shot in reconstruction["shots"]:
                orig_translation = np.array(reconstruction["shots"][shot]["translation"])
                rotation = reconstruction["shots"][shot]["rotation"]
                camera_name = reconstruction["shots"][shot]["camera"]
                qvec_T = angle_axis_to_quaternion(rotation)
                translation = orig_translation
                # translation = -orig_translation
                R_T = qvec2rotmat(qvec_T)
                qvec = rotmat2qvec(R_T)
                translation = -np.dot(R_T,orig_translation)
                x, y = 0, 0 # 画像座標 [X, Y]
                points3D_id = 0 # 3次元点のID [X, Y, Z]
                qw = qvec[0]
                qx = qvec[1]
                qy = qvec[2]
                qz = qvec[3]
                tx = translation[0]
                ty = translation[1]
                tz = translation[2]
                camera_id = cameras[camera_name]["CAMERA_ID"] # カメラのID camera_id
                name = shot # 画像の名前 name
                points2d = [x, y, points3D_id]

                # images[image_id] = {image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name, points2d} # image.txt
                images[image_id] = {
                    "IMAGE_ID": image_id,
                    "QW": qw,
                    "QX": qx,
                    "QY": qy,
                    "QZ": qz,
                    "TX": tx,
                    "TY": ty,
                    "TZ": tz,
                    "CAMERA_ID": camera_id,
                    "NAME": name,
                    "POINTS2D": points2d
                }
                image_id += 1

            for point_idx in reconstruction["points"]:
                color = reconstruction["points"][point_idx]["color"]
                coordinates = reconstruction["points"][point_idx]["coordinates"]
                xx = coordinates[0]
                yy = coordinates[1]
                zz = coordinates[2]
                r = color[0]
                g = color[1]
                b = color[2]
                error = 0

                # points3ds[point_id] = {point_id, xx, yy, zz, r, g, b, error} # points3D.txt
                points3ds[point_id] = {
                    "POINT_ID": point_id,
                    "X": xx,
                    "Y": yy,
                    "Z": zz,
                    "R": r,
                    "G": g,
                    "B": b,
                    "ERROR": error
                }
                point_id += 1
        
    return cameras, images, points3ds

def write_sparse(cameras, images, points3ds, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cameras_file = os.path.join(output_dir, 'cameras.txt')
    images_file = os.path.join(output_dir, 'images.txt')
    points3D_file = os.path.join(output_dir, 'points3D.txt')

    with open(cameras_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   camera_id, model, width, height, params[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for camera_name, elems in cameras.items():
            camera_id, model, width, height, params = elems.values()
            if(len(params) > 0):
                f.write(f"{camera_id} {model} {width} {height} {params[0]} {width/2} {height/2}\n")
            else:
                f.write(f"{camera_id} {model} {width} {height}\n")

    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name")
        f.write("#   points2d[] as (X, Y, point_id)\n")
        f.write(f"# Number of images: {len(images)}\n")
        for image_name, elems in images.items():
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name, points2d = elems.values()
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {name}\n")
            f.write(f"{points2d[0]} {points2d[1]} {points2d[2]}\n")
    
    with open(points3D_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   point_id, X, Y, Z, R, G, B, error, TRACK[] as (image_id, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points3ds)}\n")
        for point_name, elems in points3ds.items():
            point_id, x, y, z, r, g, b, error = elems.values()
            f.write(f"{point_id} {x} {y} {z} {int(r)} {int(g)} {int(b)} {error}\n")

if __name__ == "__main__":
    opensfm_dir = "/home/jaxa/shintaro/NeuS/dataset/TanksAndTemple/Auditorium"
    output_dir = os.path.join(opensfm_dir, "colmap/sparse/0")
    cameras, images, points3ds = ConvertFormat(opensfm_dir)
    write_sparse(cameras, images, points3ds, output_dir)