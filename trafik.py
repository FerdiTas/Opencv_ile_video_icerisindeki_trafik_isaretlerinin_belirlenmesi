import cv2 as cv

cam=cv.VideoCapture("levhalar5.mp4")
#cam=cv.VideoCapture(0)

parkYapmakYasak=cv.CascadeClassifier("C:\\Users\\Ferdi\\Desktop\\projem\\parkYapmakYasak\\classifier\\cascade.xml")
yayaGecidi=cv.CascadeClassifier("C:\\Users\\Ferdi\\Desktop\\projem\\yayaGecidi\\classifier\\cascade.xml")
sagaYolVar=cv.CascadeClassifier("C:\\Users\\Ferdi\\Desktop\\projem\\sagaYolVar\\classifier\\cascade.xml")
trafikIsigi=cv.CascadeClassifier("C:\\Users\\Ferdi\\Desktop\\projem\\trafikIsigi\\classifier\\cascade.xml")
sagaDonusYapilamaz=cv.CascadeClassifier("C:\\Users\\Ferdi\\Desktop\\projem\\sagaDonusYapilamaz\\classifier\\cascade.xml")
yolVer=cv.CascadeClassifier("C:\\Users\\Ferdi\\Desktop\\projem\\yolVer\\classifier\\cascade.xml")
hizSiniri50=cv.CascadeClassifier("C:\\Users\\Ferdi\\Desktop\\projem\\hizSiniri50\\classifier\\cascade.xml")
kasisVar=cv.CascadeClassifier("C:\\Users\\Ferdi\\Desktop\\projem\\kasisVar\\classifier\\cascade.xml")
okulGecidi=cv.CascadeClassifier("C:\\Users\\Ferdi\\Desktop\\projem\\okulGecidi\\classifier\\cascade.xml")
tersYon=cv.CascadeClassifier("C:\\Users\\Ferdi\\Desktop\\projem\\tersYon\\classifier\\cascade.xml")

font1=cv.FONT_HERSHEY_SIMPLEX

while True:
    ret,frame=cam.read()
    #frame=cv.flip(frame,1)
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    if parkYapmakYasak:
        levha=parkYapmakYasak.detectMultiScale(gray,1.3,15)
        for (x,y,w,h) in levha:
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv.putText(frame,"Park Yapmak Yasak!",(x,y),font1,1,(0,0,255),2)
    if yayaGecidi:
        levha2=yayaGecidi.detectMultiScale(gray,1.3,15)
        for (x1,y1,w1,h1) in levha2:
            cv.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
            cv.putText(frame,"yaya gecidi)",(x1,y1),font1,1,(0,0,255),2)
    if sagaYolVar:
        levha3=sagaYolVar.detectMultiScale(gray,1.3,13)
        for (x2,y2,w2,h2) in levha3:
            cv.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
            cv.putText(frame,"saga yol var",(x2,y2),font1,1,(0,0,255),2)
    if trafikIsigi:
        levha4=trafikIsigi.detectMultiScale(gray,1.3,10)
        for (x2,y2,w2,h2) in levha4:
            cv.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
            cv.putText(frame,"trafik isigi",(x2,y2),font1,1,(0,0,255),2)
    if sagaDonusYapilamaz:
        levha5=sagaDonusYapilamaz.detectMultiScale(gray,1.3,12)
        for (x2,y2,w2,h2) in levha5:
            cv.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
            cv.putText(frame,"Saga donus yapilamaz",(x2,y2),font1,1,(0,0,255),2)

    if yolVer:
        levha6=yolVer.detectMultiScale(gray,1.3,15)
        for (x2,y2,w2,h2) in levha6:
            cv.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
            cv.putText(frame,"yol ver",(x2,y2),font1,1,(0,0,255),2)
    if hizSiniri50:
        levha7=hizSiniri50.detectMultiScale(gray,1.3,12)
        for (x2,y2,w2,h2) in levha7:
            cv.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
            cv.putText(frame,"hiz siniri 50",(x2,y2),font1,1,(0,0,255),2)

    if kasisVar:
        levha8=kasisVar.detectMultiScale(gray,1.3,15)
        for (x2,y2,w2,h2) in levha8:
            cv.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
            cv.putText(frame,"Kasis var",(x2,y2),font1,1,(0,0,255),2)
    if okulGecidi:
        levha9=okulGecidi.detectMultiScale(gray,1.3,8)
        for (x2,y2,w2,h2) in levha9:
            cv.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
            cv.putText(frame,"Okul Gecidi",(x2,y2),font1,1,(0,0,255),2)
    if tersYon:
        levha10=tersYon.detectMultiScale(gray,1.3,13)
        for (x2,y2,w2,h2) in levha10:
            cv.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
            cv.putText(frame,"Ters Yon",(x2,y2),font1,1,(0,0,255),2)

    cv.namedWindow("Tabela",cv.WINDOW_NORMAL)
    cv.imshow("Tabela",frame)
    
    if cv.waitKey(16) & 0xFF==ord("q"):
        break
    
cam.release()
cv.destroyAllWindows()