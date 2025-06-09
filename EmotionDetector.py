from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
import cv2

# Image data generators
train_data_gen = ImageDataGenerator(rescale=1. / 255)
test_data_gen = ImageDataGenerator(rescale=1. / 255)

# Training data
train_generator = train_data_gen.flow_from_directory(
    'C:/Users/nppra/PycharmProjects/PythonProject3/archive (15)/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# Testing data
test_generator = test_data_gen.flow_from_directory(
    'C:/Users/nppra/PycharmProjects/PythonProject3/archive (15)/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# Model architecture
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Model summary
emotion_model.summary()

# Learning rate schedule
initial_learning_rate = 0.0001
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True
)

# Optimizer
optimizer = Adam(learning_rate=lr_schedule)

# Compile the model
emotion_model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

emotion_model_info=emotion_model.fit(
    train_generator,
steps_per_epoch=28709 // 64,
        epochs=30,
        validation_data=test_generator,
        validation_steps=7178 // 64
)
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights('emotion_model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(1280,720))
    if not ret:
        print(ret)
    face_detector=cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    num_faces=face_detector.detectMultiScale(gray_frame,scaleFactor=1.3,minNeighbors=5)
    for(x,y,w,h) in num_faces:
        cv2.rectangle(frame,(x,y-50),(x+w,y+h+10),(0,255,0),4)
        roi_gray_frame=gray_frame[y:y+h,x:x+w]
        cropped_img=np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame,(48,48)),-1),0)

        emotion_predict=emotion_model.predict(cropped_img)
        maxindex=int(np.argmax(emotion_predict))
        cv2.putText(frame,emotion_dict[maxindex],(x+5,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow('Emotion Detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cap.destroyAllWindows()