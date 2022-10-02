# AI_CV-basic
2022.10.01/02/08/09
20221001 오후

동영상은 한 장의 프레임이 연결된 것
YOLOv5에 넣어서 object detection 과정 살펴보기
YOLOv5는 2019년에 나옴.
pre trained model 사전학습이 되어있음
자동차:2 , 신호등:9 , 코끼리 등등 80개 이상 학습되어있음 이외에 것은 custom data로 학습시키면 됨

python data abstraction에 대한 이해
GitHub :  JSJeong-me/GSC_openCV/Data_type.ipynb 참고

가장 중요한 data type : list / set / tuple 각각의 feature가 중요
> list [중괄호] : 중복허용, 순서있음, 다른 데이터타입 삽입 가능
> set {대괄호} : 집합. 중복안됨, 순서가 없음
> tuple (소괄호) : 순서쌍. 순서나 값을 바꿀 수 없음 (고정된 값을 선언하기에 좋음)
                      tuple=(1)이라고 입력하면 int가 됨. tuple로 선언하기 위해서는 tuple=(1, ) 콤마를 넣어주어야 함

Numpy의 array (list와 헷갈리지 말기!)
import numpy as np
array는 속도가 더 빠름 array[ : , 2] 

Pandas ; 패키지 필요
import pandas as pd
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html 참고

list -> array -> dataframe(pandas) 원하는 형태로 데이터 수정하기 위해 자유자재로 변환하는 것이 중요

OpenCV download and many opensource project code
https://pypi.org/


이미지 오리기

내일
sampling할 때 내일은 동영상 프레임 오리기
초당 30프레임 정도씩 들어오는 동영상
용도에 따라 프레임 수 결정 가능 (산불감시 CCTV는 1분에 한 프레임, 주차장 차량 인식은 1초에 1프레임 등)
자율주행차가 신호등 인식 시 어떤 신호등을 기준으로 따라야하는 지 판단하도록 처리
color space 색을 분별해내는 방법 (갈변된 바나나와 싱싱한 바나나 구분하기)


20221002 오전
YOLO신경망을 활용한 이미지 인식 개념 이해
YOLO :
테두리상자 조정 (Bounding Box Coordinate)과 분류(Classification)를 
동일 신경망 구조를 통해 동시에 실행하는 통합인식(Unified Detection)을 구현

- 기존의 Object Detection은 single window나 regional proposal methods등을 통해 
바운딩 박스를 잡은 후 탐지된 바운딩 박스에 대해 분류 수행하는 2 stage detection

하나의 컨볼루션 네트워크를 통해 대상의 위치와 클래스를 한번에 예측 
(한 번만에 image detection을 할 수 있는 1 stage detection 알고리즘)
테두리상자 조정 (Bounding Box Coordinate)과 분류(Classification)를 
동일 신경망 구조를 통해 동시에 실행하는 통합인식(Unified Detection)을 구현



https://lynnshin.tistory.com/48 참고

오후 
색공간

Google Cloud Vision API 에서 RGB high, low 값 복사 

RGB(251, 208, 22)
RGB(255, 213, 5)
RGB(221, 195, 58)

	R G B
high [255, 213, 58]
low [221, 195, 5]

banana1-good
[0,32,179]
[255,255,255]

banana2-bad
[9,106,0]
[255,255,255]

signal-yellow
[12,23,237]
[255,255,255]

Frame 원래 이미지

HSV값 : 대부분 코드값으로 범위가 지정되어 나옴.
RGB -> SHV
효율적으로 색을 분별하는 방법


Object Detection 
Optimizer
Gridient Descent
2013~

다음 주
Deep Learning Tensorflow, Keras framework를 이용해서 Classification하는 모델 학습 방법
OpenCV이용한 Img Processing 과정
Custom module학습 하는 방법
CNN World
