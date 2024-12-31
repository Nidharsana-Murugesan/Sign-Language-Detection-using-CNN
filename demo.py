import cv2
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf
from keras.models import load_model
import numpy as np
import math
from tensorflow.keras.utils import img_to_array
from keras.optimizers import Adam
import time

# adam = Adam(learning_rate=0.00001)

map_characters = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm',
                  13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y',
                  25: 'z'}


associated_words = {
    'a': ["apple", "ant", "arrow", "astronaut", "anchor"],
    'b': ["ball", "bat", "bee", "banana", "book"],
    'c': ["cat", "car", "cake", "cap", "corn"],
    'd': ["dog", "duck", "doll", "drum", "door"],
    'e': ["egg", "elephant", "ear", "eye", "eagle"],
    'f': ["fish", "frog", "fan", "fork", "foot"],
    'g': ["goat", "grape", "gate", "girl", "gift"],
    'h': ["hat", "house", "horse", "hand", "hill"],
    'i': ["ice", "ink", "igloo", "iron", "island"],
    'j': ["jug", "jam", "jet", "jar", "juice"],
    'k': ["kite", "king", "key", "kid", "koala"],
    'l': ["lion", "lamp", "leaf", "lock", "leg"],
    'm': ["mouse", "monkey", "moon", "mug", "map"],
    'n': ["nose", "net", "nest", "nail", "nurse"],
    'o': ["owl", "orange", "ocean", "oven", "octopus"],
    'p': ["pen", "pig", "pot", "piano", "pearl"],
    'q': ["queen", "quilt", "quill", "quiz", "quart"],
    'r': ["rat", "rose", "ring", "rabbit", "rain"],
    's': ["sun", "star", "sock", "ship", "shoe"],
    't': ["tree", "tiger", "top", "tub", "towel"],
    'u': ["umbrella", "unicorn", "uniform", "up", "urn"],
    'v': ["van", "vase", "viper", "vest", "vase"],
    'w': ["wolf", "window", "whale", "watch", "wind"],
    'x': ["x-ray", "xylophone", "xenon", "xerox", "xenon"],
    'y': ["yak", "yam", "yard", "yarn", "yawn"],
    'z': ["zebra", "zoo", "zip", "zone", "zero"]
}


imgpath = cv2.imread(r"asl_dataset\a\hand1_a_bot_seg_1_cropped.jpeg")

def edge_detection(image):
    minValue = 70
    blur = cv2.GaussianBlur(image, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return res

def preprocessor_predict(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = edge_detection(img)
    img = img_to_array(img)
    cv2.imshow("Imgpreprocess", img)
    img = cv2.resize(img, (64, 64))
    img = img.reshape(-1, 64, 64, 1)
    return predictor(img)

def predictor(img):
    model = load_model('ASL_Predictor.h5', compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    pred = model.predict(img)
    return pred

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(maxHands=2)
offset = 15
imgSize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        predtor = preprocessor_predict(imgWhite)
        pred_classes = np.argmax(predtor, axis=1)
        predicted_char = map_characters.get(pred_classes[0])

        # Overlay the predicted character on the original image
        cv2.putText(img, predicted_char, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Create a new frame for displaying the character and associated words
        display_frame = np.ones((300, 500, 3), np.uint8) * 255
        cv2.putText(display_frame, f"Predicted: {predicted_char.upper()}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        if predicted_char in associated_words:
            for i, word in enumerate(associated_words[predicted_char]):
                cv2.putText(display_frame, word, (10, 80 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)




    cv2.imshow("Originalimg", img)
    if 'display_frame' in locals():
        cv2.imshow("Prediction and Associated Words", display_frame)


    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
