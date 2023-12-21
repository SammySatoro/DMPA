import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('/ind2/images/2.jpg')

# Преобразование изображения из BGR в HSV 2 6 7 8 9 10
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Определение диапазона цвета
lower_red_x = np.array([0, 100, 100])
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

# Нахождение контуров на изображении
contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Отображение контуров на исходном изображении
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Отображение исходного и обработанного изображений
cv2.imshow('Original Image', image)
cv2.imshow('Contours', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()