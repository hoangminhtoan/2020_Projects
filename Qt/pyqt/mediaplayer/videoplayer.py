from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

import sys

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Video Media Player")
        self.setGeometry(350, 100, 700, 500)
        self.setWindowIcon(QIcon("images/application-image.png"))

        p = self.palette()
        p.setColor(QPalette.Window, Qt.gray)
        self.setPalette(p)

        self.setupUi()
        self.show()

    def setupUi(self):
        # create media
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        # create videowidget object
        self.videoWidget = QVideoWidget()

        # create open button
        self.openButton = QPushButton("Open Video")
        self.openButton.pressed.connect(self.open_file)

        # create play button
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.pressed.connect(self.play_video)

        # create slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)

        # create hbox layout
        hboxLayout = QHBoxLayout()
        hboxLayout.setContentsMargins(0, 0, 0, 0)

        # set widgets to the hbox layout
        hboxLayout.addWidget(self.openButton)
        hboxLayout.addWidget(self.playButton)
        hboxLayout.addWidget(self.slider)

        # create vbox layout
        vboxLayout = QVBoxLayout()
        vboxLayout.addWidget(self.videoWidget)
        vboxLayout.addLayout(hboxLayout)

        self.setLayout(vboxLayout)
        self.player.setVideoOutput(self.videoWidget)

        # media player signals
        self.player.stateChanged.connect(self.mediastata_changed)
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")

        if filename != '':
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
            self.playButton.setEnabled(True)

    def play_video(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def mediastata_changed(self, state):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause)
            )
        else:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay)
            )

    def position_changed(self, position):
        self.slider.setValue(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)

    def set_position(self, position):
        self.player.setPosition(position)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()

    app.exec_()


