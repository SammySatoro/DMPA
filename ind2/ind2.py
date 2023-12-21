import os
import cv2
import numpy as np
import time

def get_image_paths():
    directory = "images"
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory)]
    return file_paths


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



def apply_sobel_operators(img):
    # ядра соболя
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    img_x = convolve(img, kernel_x)
    img_y = convolve(img, kernel_y)

    # вычисление величины градиента
    matrix_gradient = np.sqrt(img_x ** 2 + img_y ** 2)

    # вычисление углы уклона
    # Функция возвращает массив углов в радианах, где каждый угол - это арктангенс соответствующей пары элементов из y и x.
    # Углы находятся в диапазоне [-pi, pi] и измеряются против часовой стрелки от положительной оси x.
    img_angles =  np.arctan2(img_y, img_x)
    # Функция преобразует радианы в градусы
    img_angles = np.degrees(img_angles)

    # нормализация и масштабирование значений градиента
    max_gradient = np.max(matrix_gradient)
    img_gradient_to_print = (matrix_gradient / max_gradient) * 255

    img_angles_to_print = (img_angles / 360) * 255

    return img_gradient_to_print.astype(np.uint8)



def apply_canny(image, operator, gaussian_blur_kernel, low_threshold, high_threshold, standart_deviation):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (gaussian_blur_kernel, gaussian_blur_kernel), standart_deviation)

    start_time = time.time()

    if operator == 'sobel':
        edges = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 1, ksize=3)
        # edges = apply_sobel_operators(blurred_image)
    elif operator == 'prewitt':
        prewitt_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        prewitt_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(prewitt_x**2 + prewitt_y**2).astype(np.uint8)
    elif operator == 'canny':
        edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    else:
        return None

    elapsed_time = time.time() - start_time
    print(f"Время выполнения ({operator}): {elapsed_time:.4f} секунд")

    return edges, elapsed_time


def test_canny_parameters(image_paths, operator):
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        average_execution_time = 0.0
        tests_number = 0

        for blur_kernel in [3, 5, 7]:
            for lower_threshold, upper_threshold in [(50, 150), (100, 200), (150, 250)]:
                for standard_deviation in [0, 10, 100]:
                    edges, execution_time = apply_canny(image, operator, blur_kernel, lower_threshold, upper_threshold, standard_deviation)
                    if edges is not None:
                        cv2.imshow(
                            f"Parameters: Operator={operator}, Blur={blur_kernel}, Lower Threshold={lower_threshold}, Upper Threshold={upper_threshold}, Standard Deviation={standard_deviation}",
                            edges)
                        cv2.waitKey(0)

                        average_execution_time += execution_time
                        tests_number += 1

        if tests_number > 0:
            average_execution_time /= tests_number
            print(f"Среднее время выполнения для {operator} на изображении {path}: {average_execution_time:.4f} секунд")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    image_paths = get_image_paths()

    # test_canny_parameters(image_paths, operator='sobel')
    # test_canny_parameters(image_paths, operator='prewitt')
    test_canny_parameters(image_paths, operator='sobel')

