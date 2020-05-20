from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Only needed for acces to command line arguments
import sys 


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        
        self.setWindowTitle('My First Qt App')
        label = QLabel('This is awesome!!!')
        self.setCentralWidget(label)

app = QApplication(sys.argv)

window = MainWindow()
window.show()
# Start the event loop
app.exec_()