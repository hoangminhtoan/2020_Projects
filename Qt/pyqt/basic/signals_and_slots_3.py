from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys
from random import choice

window_titles = [ 'My App',
    'My App',
    'Still My App',
    'Still My App',
    'What on earth',
    'What on earth',
    'This is surprising',
    'This is surprising',
    'Something went wrong'
]

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.n_times_clicked = 0
        self.setWindowTitle("My App")

        self.btn = QPushButton("Press Me!")

        self.btn.clicked.connect(self.btnClicked)
        self.windowTitleChanged.connect(self.theWindowTitleChanged)

        self.setCentralWidget(self.btn)

    def btnClicked(self):
        print("Clicked!")
        new_window_title = choice(window_titles)
        print("Setting title: {}".format(new_window_title))
        self.setWindowTitle(new_window_title)

    def theWindowTitleChanged(self, window_title):
        print("Window title changed: {}".format(window_title))

        if "wrong" in window_title:
            self.btn.setDisabled(True)

def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec_()

if __name__ == '__main__':
    main()
