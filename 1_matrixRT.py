import cv2
import numpy as np


def show_error(img, corners, imgpts, uvErrorPath):
    """ 可视化角点像素距离误差，返回世界坐标映射得到的pred_uv和真实像素值corners 之间的误差 """
    total_error, avg_error = 0, 0
    for i in range(len(corners)):
        img = cv2.line(img, tuple(corners[i]), tuple(imgpts[i]), (0, 0, 255), 2)
        error = cv2.norm(corners[i], imgpts[i], cv2.NORM_L2) / len(corners)
        total_error += error
    avg_error = total_error / len(corners)
    print("Average Error of Reproject: ", avg_error)
    cv2.imwrite(uvErrorPath, img)
    return img, avg_error


def calib_corner_xyz(origin_point, inter_corner_shape, size_per_grid):
    """get the array of coordinates XYZ for 11*8 corners points """
    w, h = inter_corner_shape
    x_array = range(int((w - 1) / 2), int((w - 1) / 2) - w, -1)
    xyz = np.zeros((w * h, 3), np.float32)
    xyz[:, :2] = np.mgrid[x_array, 0:h].T.reshape(-1, 2)
    xyz = xyz * size_per_grid + origin_point
    return xyz


def calibration_RT(RT_imgpath, inter_corner_shape, world_point, RTMatrixPath, uvErrorPath):
    """ get R and T """
    x_nums, y_nums = inter_corner_shape
    # 设置角点查找限制，获取更精确的角点位置
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    image = cv2.imread(RT_imgpath)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 查找角点
    ret, corners = cv2.findChessboardCorners(gray, (x_nums, y_nums), )
    if ret:
        exact_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # cv2.drawChessboardCorners(image, (x_nums, y_nums), corners, ret)
        # cv2.imshow('FoundCorners', image)
        # cv2.waitKey(100)

        # 获取外参
        _, rvec, tvec, inliers = cv2.solvePnPRansac(world_point, exact_corners, mtx, dist)
        # 获得的旋转矩阵 rvec 是向量，是3×1的矩阵，想要还原回3×3的矩阵，需要罗德里格斯变换Rodrigues
        rotation_m, _ = cv2.Rodrigues(rvec)  # 罗德里格斯变换

        print('旋转矩阵是：\n', rvec)
        print('旋转矩阵3*3是：\n', rotation_m)
        print('平移矩阵是:\n', tvec)

        # 旋转矩阵和平移矩阵组成的其次矩阵
        # rotation_t = np.hstack([rotation_m, tvec])
        # rotation_t_Homogeneous_matrix = np.vstack([rotation_t, np.array([[0, 0, 0, 1]])])

        # 计算误差
        pred_uv, jac = cv2.projectPoints(world_point, rvec, tvec, mtx, dist)
        pred_uv = np.squeeze(pred_uv)
        corners = np.squeeze(corners)
        # 可视化角点像素距离误差，返回世界坐标映射得到的pred_uv和真实像素值corners 之间的误差
        img, avg_error = show_error(image, corners.astype(int), pred_uv.astype(int), uvErrorPath)

        # 保存到txt中
        f = open(RTMatrixPath, 'w')
        f.write('旋转矩阵是：\n')
        f.write(str(rvec))
        f.write("\n旋转矩阵3*3是：\n")
        f.write(str(rotation_m))
        f.write("\n平移矩阵是:\n")
        f.write(str(tvec))
        f.write("\npred_uv到corners的误差是:\n")
        f.write(str(avg_error))

        return 0


if __name__ == '__main__':
    # ---------------- initialization ---------------- #
    # 读取相机内参
    dist = np.array([[-1.49951471e-01, 5.86141377e-02, -2.86733634e-04, 4.86440964e-05, 5.39879610e-01]])
    mtx = np.array([[1.85245266e+03, 0.00000000e+00, 6.24814106e+02],
                    [0.00000000e+00, 1.85239519e+03, 5.23286923e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    inter_corner_shape = (11, 8)
    size_per_grid = 0.045
    origin_point = np.array([0, 1.4, 0])
    # 标定图像保存路径
    calib_RT_imgpath = r"data\python\RTMatrix\images\RT.tif"
    RTMatrixPath = 'result/RTMatrix.txt'
    uvErrorPath = 'result/pred_uv_error.png'

    # ---------------- calibration ---------------- #
    # 真实世界坐标
    corner_xy = calib_corner_xyz(origin_point, inter_corner_shape, size_per_grid)
    # get R and T
    calibration_RT(calib_RT_imgpath, inter_corner_shape, corner_xy, RTMatrixPath, uvErrorPath)
