import numpy as np
import cv2
import pandas as pd


def pixel_to_world(camera_intrinsics, r, t, img_points, Z0):
    K_inv = camera_intrinsics.I  # 返回矩阵的逆矩阵
    R_inv = np.asmatrix(r).I
    R_inv_T = np.dot(R_inv, np.asmatrix(t))  # 矩阵乘法
    world_points = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_point in img_points:
        coords[0] = img_point[0]
        coords[1] = img_point[1]
        coords[2] = 1.0
        cam_point = np.dot(K_inv, coords)
        cam_R_inv = np.dot(R_inv, cam_point)
        scale = (R_inv_T[2][0] + Z0) / cam_R_inv[2][0]  # Zc
        scale_world = np.multiply(scale, cam_R_inv)  # 对应元素相乘
        world_point = np.asmatrix(scale_world) - np.asmatrix(R_inv_T)
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0] = world_point[0]
        pt[1] = world_point[1]
        pt[2] = world_point[2]
        # world_points.append(pt.T.tolist())
        world_points.append(pt.T)
    world_points = np.array(world_points).squeeze()

    return world_points


def calib_corner_xyz(origin_point, inter_corner_shape, size_per_grid):
    """get the array of coordinates XYZ for 11*8 corners points """
    w, h = inter_corner_shape
    x_array = range(int((w - 1) / 2), int((w - 1) / 2) - w, -1)
    xyz = np.zeros((w * h, 3), np.float32)
    xyz[:, :2] = np.mgrid[x_array, 0:h].T.reshape(-1, 2)
    xyz = xyz * size_per_grid + origin_point
    return xyz


def show_error(img, corners, imgpts):
    # corner = tuple(corners[0].ravel())
    total_error, avg_error = 0, 0
    for i in range(len(corners)):
        img = cv2.line(img, tuple(corners[i]), tuple(imgpts[i]), (255, 0, 0), 2)
        error = cv2.norm(corners[i], imgpts[i], cv2.NORM_L2) / len(corners)
        total_error += error
    avg_error = total_error / len(corners)
    print("Average Error of Reproject: ", avg_error)

    return img, avg_error


if __name__ == '__main__':
    # ---------------- initialization ---------------- #
    camera_parameter = {
        "intrinsic": [[1.85245266e+03, 0.00000000e+00, 6.24814106e+02],
                      [0.00000000e+00, 1.85239519e+03, 5.23286923e+02],
                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
        "dist": [[-1.49951471e-01, 5.86141377e-02, -2.86733634e-04, 4.86440964e-05, 5.39879610e-01]],
        "R": [[9.99998122e-01, 1.60888703e-03, 1.08066048e-03],
              [1.68828001e-03, -4.49266345e-01, -8.93396273e-01],
              [-9.51869292e-04, 8.93396420e-01, -4.49268218e-01]],
        "T": [-0.11345296, 0.88491182, 0.18367103],
    }

    dist = camera_parameter["dist"]
    camera_intrinsic = np.asmatrix(camera_parameter["intrinsic"])
    r = camera_parameter["R"]
    t = np.asmatrix(camera_parameter["T"]).T  # 返回矩阵的转置矩阵

    # test_path = r"data\python\RTMatrix\images\RT.tif"
    test_path = r"data\python\test_XYZ\test_y1.6.tif"
    xyz_excelpath = 'result/xyz_test.xlsx'

    # ---------------- 2Dto3D ---------------- #
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    image = cv2.imread(test_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 查找角点
    ret, corners = cv2.findChessboardCorners(gray, (11, 8), )
    corners = np.squeeze(corners)
    # 2Dto3D
    Z0 = 0
    result = pixel_to_world(camera_intrinsic, r, t, corners, Z0)
    print(result)

    # ---------------- save data to excel ---------------- #
    # 保存图像中的像素点坐标 image_uv 到 excel
    data = pd.DataFrame(corners)
    writer = pd.ExcelWriter(xyz_excelpath)
    data.to_excel(writer, 'image_uv', header=False, index=False)

    # 保存预测的世界坐标 pred_xyz 到 excel
    data1 = pd.DataFrame(result)
    data1.to_excel(writer, 'pred_xyz', header=False, index=False)

    # 保存真实世界坐标 true_xyz 到 excel
    inter_corner_shape = (11, 8)
    size_per_grid = 0.045
    origin_point = np.array([0, 1.6, 0])
    corner_xyz = calib_corner_xyz(origin_point, inter_corner_shape, size_per_grid)

    data2 = pd.DataFrame(corner_xyz)
    data2.to_excel(writer, 'true_xyz', header=False, index=False)
    writer.save()
    writer.close()
