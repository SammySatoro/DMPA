import cv2
import numpy as np

def contour_object(binary_mask):
    height, width = binary_mask.shape

    result = np.zeros_like(binary_mask)

    # Проходим по каждому пикселю внутри изображения
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Если текущий пиксель - белый
            if binary_mask[i, j] == 255:
                # Проверяем вокруг текущего пикселя
                neighbors = [
                    binary_mask[i - 1, j - 1],
                    binary_mask[i - 1, j],
                    binary_mask[i - 1, j + 1],
                    binary_mask[i, j - 1],
                    binary_mask[i, j + 1],
                    binary_mask[i + 1, j - 1],
                    binary_mask[i + 1, j],
                    binary_mask[i + 1, j + 1],
                ]

                # Если хотя бы один из соседей - черный, то текущий пиксель находится на границе
                if 0 in neighbors:
                    result[i, j] = 255

    return result


# Загрузка изображения
image = cv2.imread('D:\PythonProjects\DMPA\ind2\images\\5.jpg')
image = cv2.resize(image, (640, 480))
# Преобразование изображения из BGR в HSV 2 6 7 8 9 10
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Определение диапазона цвета
lower_red_x = np.array([0, 80, 80])
upper_red_x = np.array([30, 255, 255])
lower_red_y = np.array([165, 100, 100])
upper_red_y = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red_x, upper_red_x)
mask2 = cv2.inRange(hsv, lower_red_y, upper_red_y)

mask = cv2.bitwise_or(mask1, mask2)

# Применение операций морфологии для улучшения результатов
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
contours = contour_object(mask)

# Отображение исходного и обработанного изображений
cv2.imshow('Original Image', image)
cv2.imshow('Contours', contours)
cv2.waitKey(0)
cv2.destroyAllWindows()