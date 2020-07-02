from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys 


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("My QT App")

        layout = QVBoxLayout()
        widgets = [QCheckBox,
            QComboBox,
            QDateEdit,
            QDateTimeEdit,
            QDial,
            QDoubleSpinBox,
            QFontComboBox,
            QLCDNumber,
            QLabel,
            QLineEdit,
            QProgressBar,
            QPushButton,
            QRadioButton,
            QSlider,
            QSpinBox,
            QTimeEdit]
        
        for w in widgets:
            layout.addWidget(w())
            
        
        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)


class MainWindowQLabel(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindowQLabel, self).__init__(*args, **kwargs)

        layout = QVBoxLayout()

        widget = QLabel("Hello Mates!")
        font = widget.font()
        font.setPointSize(30)
        widget.setFont(font)
        widget.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        layout.addWidget(widget)

        widget = QLabel()
        widget.setPixmap(QPixmap('combination.png'))
        widget.setScaledContents(True)

        layout.addWidget(widget)

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

def main():
    app = QApplication(sys.argv)

    window = MainWindowQLabel()
    window.show()

    app.exec_()

if __name__ == '__main__':
    main()

        