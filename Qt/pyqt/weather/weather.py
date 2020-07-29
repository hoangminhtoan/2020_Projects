from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from MainWindow import Ui_MainWindow

from datetime import datetime
import json 
import os 
import sys 
import requests 
from urllib.parse import urlencode 

OPENWEATHERMAP_API_KEY = '439d4b804bc8187953eb36d2a8c26a02'

"""
Get an API key from https://openweathermap.org/ to use with this
application.

"""

def from_ts_to_time_of_day(ts):
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%I%p").lstrip("0")

class WorkerSignals(QObject):
    '''
    Defines the signals avaiable from a running worker thread
    '''
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(dict, dict)

class WeatherWorker(QRunnable):
    '''
    Worker thread for weather updates
    '''
    signals = WorkerSignals()
    is_interrupted = False 

    def __init__(self, location):
        super(WeatherWorker, self).__init__()
        self.location = location

    @pyqtSlot()
    def run(self):
        try:
            params = dict(
                q = self.location,
                appid = OPENWEATHERMAP_API_KEY
            )

            url = 'https://samples.openweathermap.org/data/2.5/weather?%s' % urlencode(params)
            print(url)
            r = requests.get(url)
            weather = json.loads(r.text)

            # Check if we had a failure
            if weather['cod'] != 200:
                raise Exception(weather['message'])

            url = 'https://samples.openweathermap.org/data/2.5/forecast/daily?%s' % urlencode(params)
            r = requests.get(url)
            forecast = json.loads(r.text)

            self.signals.result.emit(weather, forecast)
        
        except Exception as e:
            self.signals.error.emit(str(e))

        self.signals.finished.emit()

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setupUi(self)

        self.pushButton.pressed.connect(self.update_weather)
        self.threadpoll = QThreadPool()
        self.show() 

    def alert(self, message):
        alert = QMessageBox.warning(self, "Warning", message)
    
    def update_weather(self):
        worker = WeatherWorker(self.lineEdit.text())
        print(self.lineEdit.text())
        worker.signals.result.connect(self.weather_result)
        worker.signals.error.connect(self.alert)
        self.threadpoll.start(worker)

    def weather_result(self, weather, forecasts):
        self.latitudeLabel.setText("{:.2f} °".format(weather['coord']['lat']))
        self.longitudeLabel.setText("{:.2f} °".format(weather['coord']['lon']))

        self.windLabel.setText("{:.2f} m/s".format(weather['wind']['speed']))

        self.temperatureLabel.setText("{:.1f} °C".format(weather['main']['temp']))
        self.pressureLabel.setText("{}".format(weather['main']['pressure']))
        self.humidityLabel.setText("{}".format(weather['main']['humidity']))

        self.sunriseLabel.setText(from_ts_to_time_of_day(weather['sys']['sunrise']))

        self.weatherLabel.setText("{} ({})".format(
            weather['weather'][0]['main'],
            weather['weather'][0]['description']))

        self.set_weather_icon(self.weatherIcon, weather['weather'])

        for n, forecast in enumerate(forecasts['list'][:5], 1):
            getattr(self, 'forecastTime{}'.format(n)).setText(from_ts_to_time_of_day(forecast['dt']))
            self.set_weather_icon(getattr(self, 'forecastIcon{}'.format(n)), forecast['weather'])
            getattr(self, 'forecastTemp{}'.format(n)).setText("{:.1f} °C".format(forecast['main']['temp']))

    def set_weather_icon(self, label, weather):
        label.setPixmap(
            QPixmap(os.path.join('images', "%s.png" % weather[0]['icon']))
        )

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    app.exec_()

