import cv2
import numpy as np
from math import atan2, degrees, pi
#Esta funcion detecta objetos segun los patrones de la cascada haar
def detectorHaar(img,haar,escala,minMargen,min,max):
    #Convertimos la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Activamos el detector
    #imagen, haar cascada, scale_factor=1.1, margenDeaceptacion>0, min_size=(0, 0),max_size(0,0)
    objetosDetectados = haar.detectMultiScale(gray, escala, minMargen,minSize=(min,min),maxSize=(max,max))

    #Iniciamos un bucle for para que de cada objeto que cumple con el patron
    #nos proporcione coordenadas y dibujemos rectangulos
    for (x,y,w,h) in objetosDetectados:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        print x,y
    return img

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image
    
def encontrarBordes(imagen):
    #recibe una imagen y lo transforma en escala de grises
    imagen_gris = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
    #src: matriz de entrada(1->CANAL de 8 bits) imagen de origen que debe ser una imagen de escala de grises
    #thresh: valor umbral se utiliza para clasificar los valores de pixel
    #maxval: valor maximo de umbral
    #type: tipo de umbral
    ret,umbral = cv2.threshold(imagen_gris,150,255,0)

    #encuentra los contornos en una imagen binaria
    #imagen: imagen umbral
    #almacenamiento: cv2.RETR_TREE
    #metodo: CV_CHAIN_APPROX_SIMPLE
    #offsert = (0,0)-> contornos
    _,contornos, jerarquia = cv2.findContours(umbral,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #dibujas los contornos de la imagen
    cv2.drawContours(imagen,contornos,-1,(0,255,128),2)

    for h,cnt in enumerate(contornos):
        #shape: forma de la imagen
        #tipoDato: unint8 -> entero absoluto (0-255)
        mascara = np.zeros(imagen_gris.shape,np.uint8)
        cv2.drawContours(mascara,[cnt],0,255,-1)
        #calcula color medio y toma una mascara como parametro
        media = cv2.mean(imagen,mask = mascara)
    return imagen
def reduceColor(im,n=4):

    indices = np.arange(0,256)   # List of all colors 

    divider = np.linspace(0,255,n+1)[1] # we get a divider

    quantiz = np.int0(np.linspace(0,255,n)) # we get quantization colors

    color_levels = np.clip(np.int0(indices/divider),0,n-1) # color levels 0,1,2..

    palette = quantiz[color_levels] # Creating the palette

    im2 = palette[im]  # Applying palette on image
    im2 = cv2.convertScaleAbs(im2) # Converting image back to uint8
    return im2
def eliminarRangoDeColores(img,colorMin,colorMax):
    #obtenemos los valores de los hsv
    hmin,smin,vmin = colorMin
    hmax,smax,vmax = colorMax
    #convertimos los colores a hsv
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # color hsv minimo aceptado
    lower_blue = np.array([hmin,smin,vmin])
    # color hsv maximo aceptado
    upper_blue = np.array([hmax,smax,vmax])
    # mascara a partir del rango aceptado
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # restamos a la imagen original todo lo que no este en la mascara.
    res = cv2.bitwise_and(img,img,mask= mask)
    return res
def intensidadColor(img,hMin,hMax,sMin,sMax,vMin,vMax):
  #convierte el frame al espacio de color hsv
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  #Se crea un array con las posiciones minimas y maximas
  lower=np.array([hMin,sMin,vMin])
  upper=np.array([hMax,sMax,vMax])

  #Crea una mascara en el rango de colores
  mask = cv2.inRange(hsv, lower, upper)

  #prueba de mascara resultante quitando bit a bit los pixeles
  res = cv2.bitwise_and(img,img, mask= mask)

  #fusiona dos imagenes con su grado de opacidad
  #addWeighted(img1,opacidad1,img2,opacidad2)
  salida=cv2.addWeighted(img,0.7,res,0.3,0)
  return salida

def contornos (imagen,d1,d2):
    # recibe una imagen y lo transforma en escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # src: matriz de entrada(1->CANAL de 8 bits) imagen de origen que debe ser una imagen de escala de grises
    # thresh: valor umbral se utiliza para clasificar los valores de pixel
    # maxval: valor maximo de umbral
    # type: tipo de umbral
    ret, umbral = cv2.threshold(imagen_gris, d1, d2, 0)
    # encuentra los contornos en una imagen binaria
    # imagen: imagen umbral
    # almacenamiento: cv2.RETR_TREE
    # metodo: CV_CHAIN_APPROX_SIMPLE
    # offsert = (0,0)-> contornos
    im2, contornos, jerarquia = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return im2  ,contornos

def anguloEntre(p1, p2):
    (x1,y1) = p1
    (x2, y2) = p2
    dx = x2 - x1
    dy = y2 - y1
    rads = atan2(-dy, dx)
    rads %= 2 * pi
    degs = int(degrees(rads)) %180

    return degs