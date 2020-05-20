from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from PyQt5 import uic 


Form, Window = uic.loadUiType('helloworld.ui')

app = QApplication([])
window = Window()
form = Form() 
form.setupUi(window)
window.show()
app.exec_()