from tkinter import *
from PIL import ImageTk, Image
import os


root = Tk()
root.geometry("1600x800+0+0")
root.configure(background='grey')
root.title("Sign Language Recognition System")



def createGesture():
	os.system("python /home/shivam/Desktop/sign-language-reconition-part-1/create_gesture.py")
def loadimageFiles():
	os.system("python /home/shivam/Desktop/sign-language-reconition-part-1/load_images.py")
def trainModel():
	os.system("python /home/shivam/Desktop/sign-language-reconition-part-1/cnn_keras.py")
def displayGesture():
	os.system("python /home/shivam/Desktop/sign-language-reconition-part-1/display_all_gestures.py")
def recognizeGesture():
	os.system("python /home/shivam/Desktop/sign-language-reconition-part-1/final_ubuntu.py")

background_image = ImageTk.PhotoImage(Image.open("/home/shivam/Desktop/sign-language-reconition-part-1/images.jpg"))
background_label = Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

Top = Frame(root, width=1000)
Top.pack(side='top')

#canvasObj = tk.Canvas(Top,width=200,height=45)
#canvasObj.create_text(100,25,text="Sign Language Recognition System")
#canvasObj.grid(row=0)
labelInfo = Label(Top,font=('arial',40,'bold'),text="Sign Language Recognition System",fg="black",bg="grey",bd=8,anchor='w')
labelInfo.grid(row=0)


buttonFrame = Frame(root,width=400,height=400,background="red")
buttonFrame.pack(side='left')

button_one = Button(buttonFrame,text="Create Gestures",bd=0,bg="orange",command=createGesture)
button_one.grid(row=0)
button_one.config(width=45)

button_one = Button(buttonFrame,text="Load Images",bd=0,bg="orange",command=loadimageFiles)
button_one.grid(row=1)
button_one.config(width=45)

button_two = Button(buttonFrame,text="Training The Model",bd=0,bg="orange",command=trainModel)
button_two.grid(row=2)
button_two.config(width=45)

button_two = Button(buttonFrame,text="Display All Gestures",bd=0,bg="orange",command=displayGesture)
button_two.grid(row=3)
button_two.config(width=45)

button_two = Button(buttonFrame,text="Recognize Gestures",bd=0,bg="orange",command=recognizeGesture)
button_two.grid(row=4)
button_two.config(width=45)

button_three = Button(buttonFrame,text="Quit",bd=0,bg="orange",command=root.quit)
button_three.grid(row=5)
button_three.config(width=45)



root.mainloop()
