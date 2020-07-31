from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

import os 
import sys 
import scipy.io as scio 
import cv2

class VideoCapture(QWidget):
    def __init__(self, filename, parent):
        super(VideoCapture, self).__init__()
        self.cap = cv2.VideoCapture(str(filename))
        self.video_frame = QLabel()
        print(filename)
        parent.layout.addWidget(self.video_frame)

    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGB)
        img = QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)

    def start(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000.0/30)

    def pause(self):
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        super(QWidget, self).deleteLater()


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(50, 50, 800, 600)
        self.setWindowTitle("PyTrack")

        self.setupUi()
        self.show()
    
    def setupUi(self):
        self.cap = None
        self.videoFileName = None

        # 
        self.videoWidget = QLabel()

        # create open button
        self.openButton = QPushButton("Open Video")
        self.openButton.clicked.connect(self.load_video)

        # Create play button
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play_video)

        # create hbox layout
        hboxLayout = QHBoxLayout()
        hboxLayout.addWidget(self.openButton)
        hboxLayout.addWidget(self.playButton)

        # create vbox layout
        vboxLayout = QVBoxLayout()
        vboxLayout.addWidget(self.videoWidget)
        vboxLayout.addLayout(hboxLayout)

        self.setLayout(vboxLayout)
        

    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(img)
            self.videoWidget.setPixmap(pix)
        else:
            self.timer.stop()
            self.timer.deleteLater()

    def play_video(self):
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.playButton.setEnabled(True)
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000.0 / 30)
    
    def pause_video(self):
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.setEnabled(True)
        print("Pause Video")
        self.timer.stop()

    def load_video(self):
        self.videoFileName, _  = QFileDialog.getOpenFileName(self, 'Select .mp4 Video File')
        if self.videoFileName != '':
            print(self.videoFileName)
            self.cap = cv2.VideoCapture(str(self.videoFileName))
            self.video_frame = QLabel()

        self.playButton.setEnabled(True)
            

    def closeApplication(self):
        choice = QMessageBox.question(self, 'Message','Do you really want to exit?',QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if choice == QMessageBox.Yes:
            print("Closing....")
            sys.exit()
        else:
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())