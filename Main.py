import threading
import serial, time
import cv2
import urllib2
import numpy as np
import sys
import bluetooth
import math
import VisionArtificial as vision
from collections import deque


cola = deque(["elemento 0"])
cola.popleft()
def controlador(cad):
    if (len(cad) > 0):
        if cad[0]=='P':
            pass#print "CONTROL COD P" # informar de algun evento cod P al controlador. Ej: encontro un objeto con sensor ultrasonico
        elif cad[0]=='Q':
            pass#print "CONTROL COD Q" # informar de algun evento cod Q al controlador. Ej:  tarea terminada
        controlador(cad[1:])
def broadcast(cad):
    if (len(cad) > 0):
        if cad[0]=='W':
            pass#print "BROAD COD W" # mandar un mensaje codigo 'W' a todos. EJ: W podria ser detener motores (detenerse)
        elif cad[0]=='D':
            pass#print "BROAD COD D" # mandar un mensaje codigo 'D' a todos. EJ: D podria ser encender motores (moverse)
        broadcast(cad[1:])

def agente(cad):
    if (len(cad) > 0):
        if cad[0]=='A':
            pass#print "AGENTE A" # interactuar con el agente A
        elif cad[0]=='B':
            pass#print "AGENTE B" # interactuar con el agente B
        controlador(cad[1:])

def listener():
    while True:
        pass#print("escuchando")
        time.sleep(2)
        cadena = "1W" # este mensaje sera enviado por bluetooht
        cola.append(cadena)
        cadena = "1D"  # este mensaje sera enviado por bluetooht
        cola.append(cadena)
        cadena = "2ABAB"  # este mensaje sera enviado por bluetooht
        cola.append(cadena)

def controlCola():
    while True:
        if(len(cola)):
            pass#print("Cola con mensajes ")
            msj = cola.popleft()
            if(len(msj)>1):
                if msj[0]=='0':
                    controlador(msj[1:])
                elif msj[0]=='1':
                    broadcast(msj[1:])
                elif msj[0]=='2':
                    agente(msj[1:])
                time.sleep(2)
        else:
            pass#print("Cola vacia")
            time.sleep(2)
def buscarPunto(img,lower,upper):
    RX, RY = -1, -1
    x, y, w, h = -1, -1, -1, -1
    remRango = vision.eliminarRangoDeColores(img, lower, upper)
    cv2.imshow("RANGO", remRango)
    gray = cv2.cvtColor(remRango, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 80, 240, 3)
    canny2, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        approx = cv2.approxPolyDP(contours[i], cv2.arcLength(contours[i], True) * 0.02, True)
        if (abs(cv2.contourArea(contours[i])) < 100 or not (cv2.isContourConvex(approx))):
            continue
        x, y, w, h = cv2.boundingRect(contours[i])
        if (w > 40 and h > 40):
            M = cv2.moments(contours[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            RX, RY = cX, cY
            #cv2.circle(img, (cX, cY),5, (100, 0, 255), -1)
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return RX, RY ,x, y, w, h

def puntoA(img):
    lower = np.array([75, 70, 100])
    upper = np.array([100, 255, 255])
    return buscarPunto(img,lower,upper)
def puntoB(img):
    lower = np.array([0, 0, 0])
    upper = np.array([255, 40, 255])
    return buscarPunto(img,lower,upper)
def detener(sock):
    sock.send('5')
def avanzar(sock):
    sock.send('1')

def girarDer(sock):
    sock.send('4')
def abrirpinza(sock):
    print "abrir"
    sock.send('6')
def cerrarpinza(sock):
    print "cerrar"
    sock.send('7')
def girarIzq(sock):
    sock.send('3')
def instruccion2(sock,w,h,angulo):
    detener(sock)
    angulo = int(angulo)
    print angulo
    if angulo>80 and angulo < 100:
        if(w>100):
            cerrarpinza(sock)
        avanzar(sock)
    else:
        if angulo>100:
            girarIzq(sock)
            print "izq"
        if angulo<80:
            girarDer(sock)
            print "der"

def instruccion(sock,c,v):
    if   (c == 'W' or c=='w'):#avanzar
        sock.send('1')
        print float(v) /1000
        time.sleep(float(v) /1000)
        sock.send('5')
    elif (c == 'S' or c=='s'):#retroceder
        sock.send('2')
        print float(v) / 1000
        time.sleep(float(v) / 1000)
        sock.send('5')
    elif (c == 'A' or c=='a'):#girarIzq
        sock.send('3')
        print float(v) / 1000
        time.sleep(float(v) / 1000)
        sock.send('5')
    elif (c == 'D' or c=='d'):#girarDer
        sock.send('4')
        print float(v) / 1000
        time.sleep(float(v) / 1000)
        sock.send('5')
    elif (c == 'X' or c=='x'):
        print "detener"
        pass
    elif (c == '1'):
        pass

def angle(pt1,pt2,pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1*dx2 + dy1*dy2))/math.sqrt(float((dx1*dx1 + dy1*dy1))*(dx2*dx2 + dy2*dy2) + 1e-10)

def camara():
    host = "192.168.1.105:8080"
    if len(sys.argv)>1:
        host = sys.argv[1]
    hoststr = 'http://' + host + '/video'
    print 'Streaming ' + hoststr
    stream=urllib2.urlopen(hoststr)
    bytes=''
    #IMPORTANTE LEER
    #El ancho y la altura de la captura de video se la configura en la aplicacion ip webcam, tiene que ser de tamanno de 320x240
    #caso contrario no funcionara correctamente
    cPosX = 320/2
    cPosY = 240
    scale = 1
    bd_addr = "20:16:10:31:22:51"
    port = 1
    sock = bluetooth.BluetoothSocket (bluetooth.RFCOMM)
    sock.connect((bd_addr,port))
    noActive = True
    while True:
        bytes+=stream.read(1024)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')

        if a!=-1 and b!=-1:
            abrirpinza(sock)
            jpg = bytes[a:b+2]
            bytes= bytes[b+2:]

            frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),-1)
            #lower = np.array([70, 70, 50])
            #upper = np.array([255, 250, 255])

            lower = np.array([50,90, 0])
            upper = np.array([255, 250, 255])
            resultado = vision.eliminarRangoDeColores(frame, lower, upper)
            cv2.imshow('rango', resultado)
            gray = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)

            canny = cv2.Canny(resultado,80,240,3)

            canny2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for i in range(0,len(contours)):
                approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.02,True)
                if(abs(cv2.contourArea(contours[i]))<100 or not(cv2.isContourConvex(approx))):
                    continue
                M = cv2.moments(contours[i])
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)

                if(len(approx) == 3):
                    x,y,w,h = cv2.boundingRect(contours[i])

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.line(frame, (int(cPosX), int(cPosY)), (cX, cY), (0, 0, 0), 5)
                    angulo = vision.anguloEntre((cPosX, cPosY), (cX, cY))
                    cv2.putText(frame, "a: " + angulo, (cX - 20, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    if(x>cPosX):
                        pass

                    cv2.putText(frame,'TRI',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),2,cv2.LINE_AA)

                elif(len(approx)>=4 and len(approx)<=6):
                    vtc = len(approx)
                    cos = []
                    for j in range(2,vtc+1):
                        cos.append(angle(approx[j%vtc],approx[j-2],approx[j-1]))
                    cos.sort()
                    mincos = cos[0]
                    maxcos = cos[-1]
                    x,y,w,h = cv2.boundingRect(contours[i])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.line(frame, (int(cPosX), int(cPosY)), ( cX, cY ), (0, 0, 0), 5)
                    angulo = str(vision.anguloEntre((cPosX,cPosY),(cX,cY)))
                    cv2.putText(frame, "a: "+angulo, (cX -20,cY+20 ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    if(vtc==4):
                        cv2.putText(frame,'RECT',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),2,cv2.LINE_AA)
                    elif(vtc==5):
                        cv2.putText(frame,'PENT',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),2,cv2.LINE_AA)
                    elif(vtc==6):
                        cv2.putText(frame,'HEXA',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),2,cv2.LINE_AA)
                else:

                    area = cv2.contourArea(contours[i])
                    x,y,w,h = cv2.boundingRect(contours[i])
                    print w,h

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.line(frame, (int(cPosX), int(cPosY)), (cX, cY), (0, 0, 0), 5)

                    angulo = str(vision.anguloEntre((cX, cY),(cPosX, cPosY)))
                    instruccion2(sock,w,h,angulo)
                    cv2.putText(frame, "a: " + angulo, (cX - 20, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    radius = w/2
                    if(abs(1 - (float(w)/h))<=2 and abs(1-(area/(math.pi*radius*radius)))<=0.2):
                        cv2.putText(frame,'CIRC',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),2,cv2.LINE_AA)


            cv2.imshow('frame',frame)
            #cv2.imshow('canny',canny)
            if cv2.waitKey(1) ==27:
                exit(0)
t = threading.Thread(target=listener)
t.start()
t2 = threading.Thread(target=controlCola)
t2.start()
t3 = threading.Thread(target=camara)
t3.start()
#bd_addr = "20:16:10:31:22:51"
#port = 1
#sock = bluetooth.BluetoothSocket (bluetooth.RFCOMM)
#sock.connect((bd_addr,port))
#noActive = True
#while 1:
#    tosend = raw_input()
# #   com = tosend[0:1]
#    val = tosend[1:]
#    if com != 'q':
#        if noActive :
#            instruccion(com,val)
#    else:
#        break
#sock.close()

#cola.append("elemento 4")




