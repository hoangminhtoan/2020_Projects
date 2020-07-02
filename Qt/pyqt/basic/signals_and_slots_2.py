from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys 

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Example of Signals and Slots")

        self.btn = QPushButton("Press Me!")
        self.btn.clicked.connect(self.btnClicked)

        # Set the central widget of the window
        self.setCentralWidget(self.btn)

    def btnClicked(self, s):
        self.btn.setText("You already clicked me!")
        self.btn.setEnabled(False)

        self.setWindowTitle("My Oneshot App")

def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec_()

if __name__ == '__main__':
    main()
