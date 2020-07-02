from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys 

class CustomDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(CustomDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("Hello")

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.btnBox = QDialogButtonBox(QBtn)
        self.btnBox.accepted.connect(self.accept)
        self.btnBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.btnBox)
        self.setLayout(self.layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        btn = QPushButton("Press me for a dialog!")
        btn.clicked.connect(self.onMyToolBarBtnClick)

        self.setCentralWidget(btn)

    def onMyToolBarBtnClick(self, s):
        print("click", s)

        dlg = CustomDialog(self)
        if dlg.exec_():
            print("Success!")
        else:
            print("Cancel")


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec_()

if __name__ == '__main__':
    main()

