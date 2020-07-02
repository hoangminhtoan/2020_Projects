from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("ToolBar and Menus")

        label = QLabel("Hello!")

        self.setCentralWidget(label)

        toolbar = QToolBar("My main toolbar")
        toolbar.setIconSize((QSize(16, 16)))
        self.addToolBar(toolbar)

        btn1 = QAction(QIcon("../images/bug.png"), "&Your button", self)
        btn1.setStatusTip("This is your button")
        btn1.triggered.connect(self.btnToolBarClicked)
        btn1.setCheckable(True)
        btn1.setShortcut((QKeySequence(Qt.CTRL + Qt.Key_P)))
        toolbar.addAction(btn1)

        toolbar.addSeparator()

        btn2 = QAction(QIcon("../images/bug.png"), "&Your button2", self)
        btn2.setStatusTip("This is your button2")
        btn2.triggered.connect(self.btnToolBarClicked)
        btn2.setCheckable(True)
        toolbar.addAction(btn2)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Hello"))

        toolbar.addSeparator()

        toolbar.addWidget(QCheckBox())

        self.setStatusBar((QStatusBar(self)))

        menu = self.menuBar()

        menu_file = menu.addMenu("&File")
        menu_file.addAction(btn1)
        menu_file.addSeparator()

        submenu_file = menu_file.addMenu("Submenu")
        submenu_file.addAction(btn2)

    def btnToolBarClicked(selfs, s):
        print("Click!", s)






def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec_()

if __name__ == '__main__':
    main()
