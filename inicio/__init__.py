'''
@author: Jairo Pulgarin
'''
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.image as img
import matplotlib.pyplot as plot
import pickle
import argparse
import PIL.Image
import PIL.ImageTk
import configparser

from cv2 import imshow

from tkinter import * 
from tkinter.ttk import * 
from tkinter.filedialog import *
from tkinter.constants import LEFT


# crear ventana tkinter  
root = Tk() 
root.iconbitmap("..\img\icon.ico")
root.title('CACIIA') 
root.geometry('{}x{}'.format(950, 650))
oframe=Frame(root,width=900, height=650)

oframe.pack()

#  Variables Globales

vardirec=StringVar()
vardirec.set(" ")
valorBoton="Capturar"
global camseleccionada
# Creating Menubar 
menubar = Menu(root) 
  
#Metodos



def cargaImagen():
    try:
        sizeancho,sizealto=(650, 900)
        global rutadir
        
        rutadir=askopenfilename()
        src = cv2.imread(rutadir)
        nuevotamano=cv2.resize(src, (sizealto,sizeancho ))
        cv2.imwrite("..\img\seleccion_rostros.jpg", nuevotamano)
        imgcargada = PIL.ImageTk.PhotoImage(file='..\img\seleccion_rostros.jpg')
        fondoImg.config(image=imgcargada)
        vardirec.set(str(rutadir))
        
        mainloop()
    except Exception as e:
        pass
    
def btnlimpiarcaja():
    vardirec.set(" ")
    fondoImg.config(image=frameImgfondo)
    cantPersonas.config(text=0)
    
def capturarimagen():
    #En este caso, 0 quiere decir que queremos acceder
    #a la cámara 0. Si hay más cámaras, puedes ir probando
    #con 1, 2, 3..
    cap = cv2.VideoCapture(camseleccionada)
    leido, frames = cap.read()
    
    if leido == True:
        cv2.imwrite("../img/captura.jpg", frames)
        print("Foto tomada correctamente")
        cap.release()#liberamos la cámara
        
        imgcargada = PIL.ImageTk.PhotoImage(file='..\img\captura.jpg')
        fondoImg.config(image=imgcargada)
        vardirec.set("..\img\captura.jpg")
        
        mainloop()
    else:
        def salirmg():
            wconfig.destroy()
            btnbuscarImg.config(state='normal')
        wconfig=Toplevel(oframe)
        wconfig.update_idletasks() 
        width = wconfig.winfo_width() 
        height = wconfig.winfo_height() 
        x = (wconfig.winfo_screenwidth() // 2) - (width // 2) 
        y = (wconfig.winfo_screenheight() // 2) - (height // 2) 
        wconfig.geometry('{}x{}+{}+{}'.format(200, 70, x, y))  
        #wconfig.geometry('{}x{}'.format(200, 50))
        wconfig.title("Error")
        wconfig.iconbitmap("..\img\error.ico")
        wcgeneral=Label(wconfig,text="Error al acceder a la cámara")
        wcgeneral.pack(side=TOP)
        btnSalirwc=Button(wconfig,text="Salir",width=8,command=salirmg)
        btnSalirwc.place(x=100,y=25) 
        btnbuscarImg.config(state='disable')  
        wconfig.protocol("WM_DELETE_WINDOW", salirmg)
        mainloop()
        cap.release()#liberamos la cámara
    
def buscarPersona():
       
    try:
        if(vardirec.get()!=" "):
            src = cv2.imread(vardirec.get())
            sizeancho,sizealto=(650, 900)
            
            contRostro=0
            def detectAndDisplay(frame):
                contRostro=0
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.equalizeHist(frame_gray)
                frame_gray = cv2.GaussianBlur(frame_gray, (1,1), 0)
                #-- Detecta rostros
                faces = face_cascade.detectMultiScale(frame_gray,1.1,4)
                
                for (x,y,w,h) in faces:
                    center = (x + w//2, y + h//2)
                    frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (0, 255, 0), 4)
                    contRostro=contRostro+1
                #-- Detecta rostros de perfil
                perfil = perfil_cascade.detectMultiScale(frame_gray,1.1,4)
                for (x2,y2,w2,h2) in perfil:
                    center = (x2 + w2//2, y2 + h2//2)
                    frame = cv2.ellipse(frame, center, (w2//2, h2//2), 0, 0, 360, (0, 255, 0), 4)
                    contRostro=contRostro+1
                    
                frame = cv2.resize(frame, (sizealto, sizeancho))
                cv2.imwrite("..\img\deteccion_rostros.jpg", frame)
                imgtk = PIL.ImageTk.PhotoImage(file='..\img\deteccion_rostros.jpg')
                fondoImg.config(image=imgtk)
                cantPersonas.config(text=contRostro)
                mainloop()
            face_cascade  =  cv2 . CascadeClassifier ( '..\Data\haarcascades_cuda\haarcascade_frontalface_alt.xml' ) 
            perfil_cascade  =  cv2 . CascadeClassifier ( '..\Data\haarcascades_cuda\haarcascade_profileface.xml' )
                    
            detectAndDisplay(src)
        else:
            def salirmg():
                wconfig.destroy()
                btnInicarCont.config(state='normal')
            wconfig=Toplevel(oframe)
            wconfig.update_idletasks() 
            width = wconfig.winfo_width() 
            height = wconfig.winfo_height() 
            x = (wconfig.winfo_screenwidth() // 2) - (width // 2) 
            y = (wconfig.winfo_screenheight() // 2) - (height // 2) 
            wconfig.geometry('{}x{}+{}+{}'.format(250, 70, x, y))  
            #wconfig.geometry('{}x{}'.format(250, 50))
            wconfig.iconbitmap("..\img\error.ico")
            wconfig.title("Error")
            wcgeneral=Label(wconfig,text="Seleccione una imagen jpg, png o camara")
            wcgeneral.pack(side=TOP)
            btnSalirwc=Button(wconfig,text="Salir",width=8,command=salirmg)
            btnSalirwc.place(x=100,y=25)   
            btnInicarCont.config(state='disable')
            wconfig.protocol("WM_DELETE_WINDOW", salirmg)
            mainloop()
            
        
    except Exception as e:
        pass


# cargar configuracián básica
configuracion = configparser.ConfigParser()
configuracion.read('saveconfig.cfg')
camseleccionada=int(configuracion['Camara']['Seleccion'])
camseleccionada=camseleccionada-1

def ventanaconfig():
    def salirwc():
        wconfig.destroy()
        btnConfugurar.config(state='normal')
    def saveconfig():
        configuracion = configparser.ConfigParser()
        # Se añade la sección 'General'
        
        configuracion['General'] = {}
        
        # Se añade la sección 'Camara' 
        
        configuracion['Camara'] = {}
        configuracion['Camara']['Seleccion'] = selecamara.get()
        camselect=int(selecamara.get())
        camseleccionada=camselect-1
        return camseleccionada
        # Guardar en un archivo la configuración
        with open('saveconfig.cfg', 'w') as archivoconfig:
            configuracion.write(archivoconfig)
        mainloop()
        
    
            
    wconfig=Toplevel(oframe)
    wconfig.geometry('{}x{}'.format(400, 400))
    wconfig.title("Configuración")
    wcgeneral=Label(wconfig,text="------Configuración General-------")
    wcgeneral.pack(side=TOP)
    wccamara=Label(wconfig,text="------Configuración Camara-------")
    wccamara.pack(side=TOP)
    wccamaraselect=Label(wconfig,text="Camara: ")
    wccamaraselect.place(x=10,y=40)
    
    selecamara=Combobox(wconfig, state="readonly", values=[1,2,3,4,5],width=15)
    selecamara.current(0)
    selecamara.place(x=60,y=40)
    labelnota=Label(wconfig,text="Nota: La selección de camaras va desde 1 en adelante")
    labelnota.place(x=10,y=100)
    labelnota=Label(wconfig,text="puede ser la camara integrada o externas")
    labelnota.place(x=10,y=120)
    btnguardarwc=Button(wconfig,text="Guardar",width=8,command=saveconfig)
    btnguardarwc.place(x=10,y=340)
    
    btnSalirwc=Button(wconfig,text="Salir",width=8,command=salirwc)
    btnSalirwc.place(x=100,y=340)
    
    btnConfugurar.config(state='disable')
    wconfig.protocol("WM_DELETE_WINDOW", salirwc)
    mainloop()

# mostrar opciones

frameImagenmenu= LabelFrame(oframe,relief=RAISED,background="#D6D6D6")
frameImagenmenu.grid(row=0,column=0,rowspan=4,sticky=S+N+E+W)
frameImagenmenu.config(width=100, height=600)    

labellibre=Label(frameImagenmenu,text=" ",height=1,background="#D6D6D6")
labellibre.grid(row=0,column=1)

def cbSeleccionFunc(eventObject):
    if(cbSeleccion.get()=="Camara"):
        btnbuscarImg.config(text="Capturar",command=capturarimagen)
        btnlimpiarcaja()
        mainloop()
    else:
        btnbuscarImg.config(text="Buscar",command=cargaImagen)
        btnlimpiarcaja()
        mainloop()
    

cbSeleccion=Combobox(frameImagenmenu,values=["Camara","Imagen"],width=8,state="readonly")
cbSeleccion.current(0)
cbSeleccion.grid(row=1,column=0)
cbSeleccion.bind("<<ComboboxSelected>>",cbSeleccionFunc)



labellibre=Label(frameImagenmenu,text=" ",height=1,background="#D6D6D6")
labellibre.grid(row=2,column=1)

btnbuscarImg=Button(frameImagenmenu,text=valorBoton,width=8,command=capturarimagen)
btnbuscarImg.grid(row=3,column=0)

labellibre=Label(frameImagenmenu,text=" ",height=1,background="#D6D6D6")
labellibre.grid(row=4,column=1)

# mostrar direccion de la imagen
lbldirecion=Label(frameImagenmenu,text="Direccion:",background="#D6D6D6")
lbldirecion.grid(row=5,column=0)

labellibre=Label(frameImagenmenu,text=" ",height=1,background="#D6D6D6")
labellibre.grid(row=6,column=1)

direcion=Entry(frameImagenmenu,state="readonly",textvariable=vardirec)
direcion.grid(row=7,column=0)

labellibre=Label(frameImagenmenu,text=" ",height=1,background="#D6D6D6")
labellibre.grid(row=8,column=1)

# Limpiesa dela imagen seccionada y baciar campos de texto
btnlimpiarImg=Button(frameImagenmenu,text="Limpiar",width=8,command=btnlimpiarcaja)
btnlimpiarImg.grid(row=9,column=0)

labellibre=Label(frameImagenmenu,text=" ",height=1,background="#D6D6D6")
labellibre.grid(row=10,column=1)

# Iniciar el condeo de personas
btnInicarCont=Button(frameImagenmenu,text="Iniciar",width=8,command=buscarPersona)
btnInicarCont.grid(row=11,column=0)
btnInicarCont.config(pady=5, padx=10)

labellibre=Label(frameImagenmenu,text=" ",height=1,background="#D6D6D6")
labellibre.grid(row=12,column=1)

labellibre=Label(frameImagenmenu,text=" ",height=20,background="#D6D6D6")
labellibre.grid(row=13,column=1)

btnConfugurar=Button(frameImagenmenu,text="Configurar",width=8,command=ventanaconfig)
btnConfugurar.grid(row=14,column=0)


#mostrar imagen de fondo y label de conteo
frameImagen= LabelFrame(oframe,relief=RAISED)
frameImagen.grid(row=2,column=1)
img=PIL.Image.open('../img/features1.png')
frameImgfondo = PIL.ImageTk.PhotoImage(img)
fondoImg=Label(frameImagen,image=frameImgfondo,width=800, height=570)
fondoImg.grid(row=0,column=0)
cantPersonaslabel=Label(oframe,text="Cantidad de Personas")
cantPersonaslabel.grid(row=0,column=1)
cantPersonas=Label(oframe,text=0, relief=GROOVE,width=10)
cantPersonas.grid(row=1,column=1)







mainloop() 