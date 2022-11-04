import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True,min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# ランドマークの座標を取得する
def face(results, annotated_image):
    label = ["x", "y", "z"]
    data = []
    if results.face_landmarks:
        # ランドマークを描画する
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

        for landmark in results.face_landmarks.landmark:
            data.append([landmark.x, landmark.y, landmark.z])

    else:  # 検出されなかったら欠損値nanを登録する
        data.append([np.nan, np.nan, np.nan])

    return data

def landmark(image):
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    annotated_image = image.copy()

    df_xyz = face(results, annotated_image)
    return df_xyz

RIGHT_EYE=[33,7,163,144,145,153,154,155,246,161,160,159,158,157,173,133]
LEFT_EYE=[263,249,390,373,374,380,381,382,466,388,387,386,385,384,398,362]
#最初(0)が外側，最後(15)が内側，下4，上11
FACE_OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
#上0 下18 右9 左27

path="test_img/unit_test/canmake4/A_in.png"
image = cv2.imread(path)
df_xyz = landmark(image)

h,w,c=image.shape
for i,cod in enumerate(RIGHT_EYE):
    y,x=(int(df_xyz[cod][0]*h),int(df_xyz[cod][1]*w))
    image[x-5:x+5,y-5:y+5,:]=0
    #cv2.imwrite("tmp/"+str(i)+".png",image)
import numpy as np
df_xyz=np.array(df_xyz)
cod=np.array([df_xyz[x] for x in RIGHT_EYE]).mean(axis=0)[:2]
#cod=df_xyz[RIGHT_EYE[0]]-df_xyz[RIGHT_EYE[15]]
y,x=int(cod[0]*h),int(cod[1]*w)
image[x-5:x+5,y-5:y+5,:]=0
cv2.imwrite("tmp/ss.png",image)
x=np.array((h,w))
print(cod*x)