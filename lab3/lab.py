import cv2
import numpy as np


size = (640, 480)
kernel_size = 5

def task():
    img = cv2.imread("/home/sammysatoro/PycharmProjects/DMPA/lab3/images/chebupizza.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    standard_deviation = 100

    img_copy = img.copy()

    img_copy = convolve(img_copy, normalize_kernel(gaussian_kernel(kernel_size, 1)))
    blurred_image = cv2.GaussianBlur(img, (5, 5), standard_deviation)
    cv2.imshow("blurred image", blurred_image)
    print(len(img_copy))
    print(len(img_copy[0]))
    cv2.imshow("orig image", img)
    cv2.imshow("my blur image", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2 * sigma**2)),
        (size, size)
    )
    return kernel

def normalize_kernel(kernel):
    return kernel / np.sum(kernel)


def convolve(orig_matrix, kernel):
    start = kernel_size // 2
    submatrices = get_orig_matrix_areas(orig_matrix)

    for i in range(len(submatrices)):
        submatrices[i] = np.sum(np.multiply(submatrices[i], kernel))

    convolved_size = (size[0] - kernel_size + 1, size[1] - kernel_size + 1)

    # for i in range(convolved_size[1]):
    #     for j in range(convolved_size[0]):
    #         print(int(submatrices[j + (i * convolved_size[1])]), end=" ")
    #     print()

    for i in range(convolved_size[1]):
        for j in range(convolved_size[0]):
            orig_matrix[start + i][start + j] = submatrices[j + i * convolved_size[0]]

    return orig_matrix


def get_orig_matrix_areas(orig_matrix):
    submatrices = []
    for i in range(len(orig_matrix) - kernel_size + 1):
        for j in range(len(orig_matrix[0]) - kernel_size + 1):
            submatrix = orig_matrix[i:i + kernel_size, j:j + kernel_size]
            submatrices.append(submatrix)
    return submatrices


task()