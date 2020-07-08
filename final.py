import dlib as dl
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from tkinter import *
import cv2 as cv


# Variable that stores number of question to be asked

no_of_que = 2

#declaring the final dataset array or training and question


final_df = np.empty((0,137),int)
que_df = np.empty((0,136),int)

#Loading the required files of dlib inside the program file
# loading the LogisticRegression model
detector = dl.get_frontal_face_detector()
predictor = dl.shape_predictor("shape_predictor_68_face_landmarks.dat")
model = LogisticRegression()




def display_tkWindow():
    global no_of_que
    frame = Tk()
    frame.title("Lie Detection")
    que_label = Label(frame, text="Number of Questions 10< ")
    que_label.pack()

    def getValue():
        global no_of_que
        no_of_que = int(e1.get())
        frame.destroy()

    q = IntVar()
    e1 = Entry(frame, textvariable=q)
    e1.pack()
    q.set(10)
    e1.delete(0, END)

    b = Button(text="Enter", command=getValue)
    b.pack()
    frame.mainloop()
    if (no_of_que < 10):
        no_of_que = 10


def train_model(df):
    global model
    X = df.loc[:, df.columns != 'target'].values
    y = df['target'].values
    print(X)
    print(y)
    model.fit(X,y)

def predict_answer():
    global model
    ans = model.predict(que_df)
    if(ans):
        print("You are safe.....\n You answered Correctly to the asnked Questions......")
    else:
        print("You are Caught......\n You answered Wrong to the asked Question")



class LieDtection:
    np_array = np.empty((0, 136), int)
    global final_df
    global que_df
    def minimixe_dataset(self,df,isQue):
        global final_df
        global que_df
        lis = np.array([])
        for i in df.T:
            lis = np.append(lis, int(np.sqrt(np.mean(i ** 2))))
        if(isQue):
            que_df = np.append(que_df , np.array(lis))
            que_df = np.array(que_df , dtype = np.int)[np.newaxis]
        else:
            ans = int(input("Answer is true or false (1 or 0) :-  "))
            lis = np.append(lis, ans)
            lis = np.array(lis, dtype=np.int)[np.newaxis]
            final_df = np.append(final_df, np.array(lis), axis=0)

    def store_dataset(self,lis):
        global np_array
        #print(lis)
        self.np_array = np.append(self.np_array, lis, axis=0)

    def make_dataset(self,landmarks):
        lis = np.array([])[np.newaxis]
        for i in range(68):
            lis = np.append(lis, [[landmarks.part(i).x]], axis=1)
            lis = np.append(lis, [[landmarks.part(i).y]], axis=1)
        lis = lis.astype('int32')
        print(lis.shape)
        self.store_dataset(lis)

    def capture_video(self , isQue = 0):
        global np_array
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        detector = dl.get_frontal_face_detector()
        predictor = dl.shape_predictor("shape_predictor_68_face_landmarks.dat")
        frame_number = 0
        while True:
            _, frame = cap.read()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = detector(gray)
            cv.rectangle(frame, (200, 130), (420, 350), (0, 0, 255), 3)
            for face in faces:
                landmark = predictor(gray, face)

                if (frame_number < 30):
                    self.make_dataset(landmark)
                elif (frame_number == 30):
                    self.minimixe_dataset(self.np_array , isQue)
                    print("Question Dataset is prepared")
                frame_number += 1
                for i in range(0, 68):
                    x = landmark.part(i).x
                    y = landmark.part(i).y
                    cv.circle(frame, (x, y), 2, (255, 0, 0), -1)
            cv.imshow("Capturing", frame)

            key = cv.waitKey(1)
            if key == ord("q"):
                break

display_tkWindow()
print("[INFO]  ", no_of_que , "Questions Will Be asked......")
for i in range(1,no_of_que+1):
    print("Question number : " , i , "should proceed now.....")
    lie = LieDtection()
    lie.capture_video()
cols = [i for i in  range(0,136)]
cols.append('target')
df = pd.DataFrame(final_df , columns = cols)
train_model(df)

print("Now Its Predicting Question Time..... \nBBBBOOOOMMMM......")
que = LieDtection()
que.capture_video(1)
predict_answer()