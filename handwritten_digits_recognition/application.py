import pygame
import sys
import cv2
import numpy as np
import math
import tensorflow as tf

# Load the digit recognition model
model = tf.keras.models.load_model('model/digit_recognition_model.keras')

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 640

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create screen
screen = pygame.display.set_mode((SCREEN_WIDTH * 2, SCREEN_HEIGHT))
pygame.display.set_caption('Draw Numbers')

# Utility functions
def save_screen(screen, filename='images/drawing.png'):
    pygame.image.save(screen, filename)
    print(f"Screen saved as {filename}")

def show_output_image(img):
    surf = pygame.pixelcopy.make_surface(img)
    surf = pygame.transform.rotate(surf, -270)
    surf = pygame.transform.flip(surf, 0, 1)
    screen.blit(surf, (SCREEN_WIDTH, 0))

def crop(original):
    cropped = pygame.Surface((SCREEN_WIDTH - 5, SCREEN_HEIGHT - 5))
    cropped.blit(original, (0, 0), (0, 0, SCREEN_WIDTH - 5, SCREEN_HEIGHT - 5))
    return cropped

# Image processing functions
def image_refiner(gray):
    org_size = 22
    img_size = 28
    rows, cols = gray.shape

    if rows > cols:
        factor = org_size / rows
        rows = org_size
        cols = int(round(cols * factor))
    else:
        factor = org_size / cols
        cols = org_size
        rows = int(round(rows * factor))
    
    gray = cv2.resize(gray, (cols, rows))
    
    cols_padding = (int(math.ceil((img_size - cols) / 2.0)), int(math.floor((img_size - cols) / 2.0)))
    rows_padding = (int(math.ceil((img_size - rows) / 2.0)), int(math.floor((img_size - rows) / 2.0)))
    
    gray = np.lib.pad(gray, (rows_padding, cols_padding), 'constant')
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return thresh

def put_label(t_img, label, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    l_x = int(x) - 7
    l_y = int(y) - 7
    cv2.rectangle(t_img, (l_x, l_y + 5), (l_x + 35, l_y - 35), (0, 255, 0), -1) 
    cv2.putText(t_img, str(label), (l_x, l_y), font, 1.5, (255, 0, 0), 1, cv2.LINE_AA)
    return t_img

def predict_digit(img):
    img = img / 255
    test_image = img.reshape(-1, 28, 28, 1)
    return np.argmax(model.predict(test_image))

# Function to detect contours and draw bounding boxes using OpenCV
def predict_numbers(filename='images/drawing.png'):
    image = cv2.imread(filename)
    if image is None:
        print(f"Error loading {filename}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        roi = cv2.bitwise_not(roi)
        rf = image_refiner(roi)
        digit = predict_digit(rf)
        put_label(image, digit, x, y)
        
    cv2.imwrite("images/predict.jpg", image)
    return image

# Main loop
def main():
    drawing = False
    last_pos = None
    screen.fill(WHITE)

    while True:
        pygame.draw.line(screen, BLACK, [SCREEN_WIDTH, 0], [SCREEN_WIDTH, SCREEN_HEIGHT], 8)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
                last_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                last_pos = None
            elif event.type == pygame.MOUSEMOTION and drawing:
                current_pos = event.pos
                if last_pos is not None:
                    pygame.draw.line(screen, BLACK, last_pos, current_pos, 5)
                last_pos = current_pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    cropped = crop(screen)
                    save_screen(cropped)
                    pred = predict_numbers()
                    show_output_image(pred)
                    save_screen(screen, "images/both.png")
                elif event.key == pygame.K_x:
                    screen.fill(WHITE)
                    pygame.draw.line(screen, BLACK, [SCREEN_WIDTH, 0], [SCREEN_WIDTH, SCREEN_HEIGHT], 8)
                    
        pygame.display.flip()

if __name__ == "__main__":
    main()
