#import io                                                         # 파일 경로 접근시 필요한 python 내장 라이브러리
# from PIL import Image                                            # Python Imaging Library. 파이썬 인터프리터에 다양한 이미지 파일 형식을 지원,
                                                                # 강력한 이미지처리와 그래픽 기능 제공 오픈소스 라이브러리
#import numpy as np
#from flask import Flask                                            # python web framework
#from flask import render_template, redirect, url_for, request    # flask에서 필요한 모듈
#from flask import jsonify                                        # import JSON을 해도되지만 여기서는 flask 내부에서 지원하는 jsonify를 사용

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D
# from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.utils import Sequence                                # 이 모듈이 없으면 사용자가 만든 generator에서 'shape'가 없다고 하는 에러가 발생할 수 있음
# import matplotlib.pyplot as plt
#import requests
from tensorflow import keras
import cv2
import numpy as np
import os
import sys
import playsound
from gtts import gTTS
from datetime import datetime
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route("/test")
def test():
    # 음성파일 저장 폴더의 파일들 제거
    if os.path.exists('./static/voice'):
        for file in os.scandir('./static/voice'):
            os.remove(file.path)

    cap = cv2.VideoCapture(0)  # 카메라 지정

    if not cap.isOpened():  # 카메라가 제대로 불러지지 않으면 오류메세지 출력
        print('video capture failed')
        sys.exit()

    while True:
        ret, frame = cap.read()  # 카메라 읽기

        if not ret:  # 읽기 실패 시 오류 메세지 출력
            print('videos read failed')
            break

        cv2.imshow("camera", frame)  # 읽어온 카메라 영상을 출력

        if cv2.waitKey(20) == ord('s'):  # s를 누르면 해당 순간의 이미지를 저장하고 종료
            cv2.imwrite("./static/images/photo.jpg", frame)
            break

    cap.release()
    cv2.destroyAllWindows()  # 카메라 및 창 닫기

    img = cv2.imread('./static/images/photo.jpg')  # 저장한 이미지 불러오기
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 타입변경
    img = cv2.resize(img, (224, 224)) / 255.0  # 모델에 맞는 input_shape로 리사이즈
    img = img.reshape((1,) + img.shape)  # 입력 데이터로 사용하기 위해 데이터 reshape
    # 모델 파일 불러와서 예측 수행
    model = keras.models.load_model('./model/VGG16_4_18_0.0300.h5')
    pred = model.predict(img)

    # 인덱스로 상품명 추출
    class_dict = {0:'갈아만든배',
                 1:'레쓰비',
                 2:'마운틴듀',
                 3:'밀키스',
                 4:'스프라이트',
                 5:'칠성사이다',
                 6:'코카콜라',
                 7:'트로피카나망고',
                 8:'펩시콜라',
                 9:'환타오렌지'}

    pred_class = class_dict[np.argmax(pred, axis=1)[0]]

    # 음성 출력부
    def save(self, filename):
        with open(filename, "wb") as file:
            file.write(self.resp.content)

    date_string = datetime.now().strftime("%d%m%Y%H%M%S")
    filename = "./static/voice/voice"+date_string+".mp3"  # 출력할 파일명 지정
    filepath = 'voice/voice'+date_string+".mp3"
    tts = gTTS(text=f'이 음료는 {pred_class}입니다', lang='ko', slow=False)
    tts.save(filename)  # 음성 파일 저장
    playsound.playsound(filename)  # 음성 출력

    return render_template('results.html', value = pred_class,
                           image_file = 'images/photo.jpg',
                           audio_file = filepath
                           )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)