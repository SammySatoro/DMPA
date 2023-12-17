import cv2
import numpy as np


def task(path, standard_deviation, kernel_size, bound_path):
    # 1
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (640, 480))
    imgBlurByCV2 = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)
    cv2.imshow(path, imgBlurByCV2)
    # 2
    matrix_gradient, img_angles, max_gradient = apply_sobel_operators(img)
    # 3
    img_border_no_filter = apply_non_max_suppression(img, img_angles, matrix_gradient)
    # 4
    apply_gradient_filter(img, matrix_gradient, img_border_no_filter, max_gradient, bound_path)

    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break

def convolve(img, kernel):
    kernel_size = len(kernel)

    x_start = kernel_size // 2
    y_start = kernel_size // 2

    matrix = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            matrix[i][j] = img[i][j]

    for i in range(x_start, len(matrix)-x_start):
        for j in range(y_start, len(matrix[i])-y_start):
            val = 0
            for k in range(-(kernel_size//2), kernel_size//2+1):
                for l in range(-(kernel_size//2), kernel_size//2+1):
                    val += img[i + k][j + l] * kernel[k + (kernel_size//2)][l + (kernel_size//2)]
            matrix[i][j] = val

    return matrix


def get_number_of_angles(x, y):
    tg = y / x if x != 0 else 127

    if x < 0:
        quadrant = 0 if y < 0 else 1
    else:
        quadrant = 2 if y < 0 else 3

    angle_lookup = {
        (0, 0): 127,
        (0, 1): 2,
        (0, 2): 3,
        (1, 0): 6,
        (1, 1): 7,
        (1, 2): 4,
        (2, 0): 1,
        (2, 1): 0,
        (2, 2): 5,
        (3, 0): 2,
        (3, 1): 3,
        (3, 2): 4,
    }

    return angle_lookup[(quadrant, int(abs(tg) > 2.414 or abs(tg) < 0.414))]

def apply_sobel_operators(img):
    # ядра соболя
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    img_x = convolve(img, kernel_x)
    img_y = convolve(img, kernel_y)

    # вычисление величины градиента
    matrix_gradient = np.sqrt(img_x ** 2 + img_y ** 2)

    # вычисление углы уклона
    img_angles = np.arctan2(img_y, img_x)
    img_angles = np.degrees(img_angles)

    # нормализация и масштабирование значений градиента
    max_gradient = np.max(matrix_gradient)
    img_gradient_to_print = (matrix_gradient / max_gradient) * 255

    img_angles_to_print = (img_angles / 360) * 255

    cv2.imshow('Gradient Magnitude', img_gradient_to_print.astype(np.uint8))
    cv2.imshow('Gradient Angle', img_angles_to_print.astype(np.uint8))

    return matrix_gradient, img_angles, max_gradient


def apply_non_max_suppression(img, img_angles, matrix_gradient):
    img_border_no_filter = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            angle = img_angles[i][j]
            gradient = matrix_gradient[i][j]
            # находится ли текущий пиксель на грани изображения(на границе матрицы)
            if i == 0 or i == img.shape[0] - 1 or j == 0 or j == img.shape[1] - 1:
                img_border_no_filter[i][j] = 0
            else:
                if angle % 2 == 0:
                    x_shift = 0
                elif 0 < angle < 4:
                    x_shift = 1
                else:
                    x_shift = -1

                if angle % 4 == 2:
                    y_shift = 0
                elif 2 < angle < 6:
                    y_shift = -1
                else:
                    y_shift = 1

                # является ли текущий пиксель локальным максимумом градиента в направлении угла
                is_max = gradient >= matrix_gradient[i + y_shift][j + x_shift] and gradient >= matrix_gradient[i - y_shift][
                    j - x_shift]
                img_border_no_filter[i][j] = 255 if is_max else 0

    cv2.imshow('Non-Maximal Suppression', img)
    return img_border_no_filter


def apply_gradient_filter(img, matr_gradient, img_border_no_filter, max_gradient, bound_path):
    # вычисление границ
    lower_bound = max_gradient / bound_path
    upper_bound = max_gradient - max_gradient / bound_path
    img_border_filter = np.zeros(img.shape, dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gradient = matr_gradient[i][j]

            if img_border_no_filter[i][j] == 255:
                if lower_bound <= gradient <= upper_bound:
                    # проверка соседних пикселей на наличие более высокого градиента
                    if any(img_border_no_filter[i + k][j + l] == 255 and matr_gradient[i + k][j + l] >= lower_bound for
                           k in range(-1, 2) for l in range(-1, 2)):
                        img_border_filter[i][j] = 255
                elif gradient > upper_bound:
                    img_border_filter[i][j] = 255

    cv2.imshow('Double Thresholding', img_border_filter)
    return img_border_filter


# 5
#task('test.jpg', 10, 3, 15)
task('D:\PythonProjects\DMPA\lab4\images\chebupizza.png', 100, 5, 6)
#task('test.jpg', 100, 11, 15)