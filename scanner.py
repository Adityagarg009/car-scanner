import cv2 as cv
cascade=("content\cars.xml")
vid=("content\h.mp4")
cap=cv.VideoCapture(vid)
fgbg=cv.createBackgroundSubtractorMOG2()
car_cas=cv.CascadeClassifier(cascade)
while True:
    ret,img=cap.read()
    fgbg.apply(img)
    if(type(img)==type(None)):
        break
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cars=car_cas.detectMultiScale(gray,5.5,1)
    for(x,y,w,h) in cars:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,225,255),2)
    cv.imshow("video",img)
    if cv.waitKey(33)==27:
        break
cv.destroyAllWindows()