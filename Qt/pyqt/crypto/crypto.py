from datetime import datetime, timedelta, date
from itertools import cycle
import os
import sys
import traceback

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import numpy as np

import pyqtgraph as pg
import requests
import requests_cache

# CryptoCompare.com API Key
CRYPTOCOMPARE_API_KEY = ''

# Define a requests http cache to minimize API requests
requests_cache.install_cache(os.path.expanduser('~/.goodforbitcoin'))

# Base currency is used to retrieve rates from bitcoinaverage
DEFAULT_BASE_CURRENCY = 'USD'
AVAILABLE_BASE_CURRENCY = ['USD', 'EUR', 'GBP']

# The crypto currencies to retrieve data about
AVAILABLE_CRYPTO_CURRENCIES = ['BTC', 'ETH', 'LTC', 'EOS', 'XRP', 'BCH']
DEFAULT_DISPLAY_CURRENCIES = ['BTC', 'ETH', 'LTC']

# Number of historic timepoints to plot (days)
NUMBER_OF_TIMEPOINTS = 150

# Color cycle to use for plotting currencies
BREWER12PAIRED = cycle(['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00',
                  '#cab2d6', '#6a3d9a', '#ffff99', '#b15928' ])

# Base PyQtGraph configuration
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class WorkerSignals(QObject):
    """
    Defines the signals available from a running working thread
    """
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    progress = pyqtSignal(int)
    data = pyqtSignal(dict, list)
    cancel = pyqtSignal()

class UpdateWorker(QRunnable):
    """
    Worker thread fro update currency
    """
    signals = WorkerSignals()

    def __int__(self, base_currency):
        super(UpdateWorker, self).__init__(base_currency)

        self.is_interrupted = False
        self.base_currency = base_currency
        self.signals.cancel.connect(self.cancel)

    @pyqtSlot()
    def run(self):
        auth_header = {
            'Apikey': CRYPTOCOMPARE_API_KEY
        }
        try:
            rates = {}
            for n, crypto in enumerate(AVAILABLE_CRYPTO_CURRENCIES, 1):
                url = 'https://min-api.cryptocompare.com/data/histoday?fsym={fsym}&tsym={tsym}&limit={limit}'
                r = requests.get(
                    url.format(**{
                    'fsym': crypto,
                    'tsym': self.base_currency,
                    'limit': NUMBER_OF_TIMEPOINTS - 1,
                    'extraParams': 'www.learnpyqt.com',
                    'format': 'json',
                    }),
                    headers=auth_header,
                )
                r.raise_for_status()
                rates[crypto] = r.json().get('Data')

                self.signals.progress.emit(int(100 * n / len(AVAILABLE_CRYPTO_CURRENCIES)))

                if self.is_interrupted:
                    return

            url = 'https://min-api.cryptocompare.com/data/exchange/histoday?tsym={tsym}&limit={limit}'
            r = requests.get(
                url.format(**{
                    'tsym': self.base_currency,
                    'limit': NUMBER_OF_TIMEPOINTS - 1,
                    'extraParams': 'www.learnpyqt.com',
                    'format': 'json',
                }),
                headers=auth_header,
            )
            r.raise_for_status()
            volume = [d['volume'] for d in r.json().get('Data')]

        except Exception as e:
            self.signals.error.emit((e, traceback.format_exc()))
            return

        self.signals.data.emit(rates, volume)
        self.signals.finished.emit()

    def cancel(self):
        self.is_interrupted = True
    
class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        layout = QHBoxLayout()

        self.ax = pg.PlotWidget()
        self.ax.showGrid(True, True)

        self.line = pg.InfiniteLine(
            pos=-20,
            pen=pg.mkPen('k', width=3),
            movable=False # We have our own code to handle dragless moving.
        )

        self.ax.addItem(self.line)
        self.ax.setLabel('left', text='Rate')
        self.p1 = self.ax.getPlotItem()
        self.p1.scene().sigMouseMoved.connect(self.mouse_move_handler)

        # Add the right-hand axis for the market activity
        self.p2 = pg.ViewBox()
        self.p2.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.p1.showAxis('right')
        self.p1.scene().addItem(self.p2)
        self.p2.setXLink(self.p1)
        self.ax2.linkToView(self.p2)
        self.ax2.setGrid(False)
        self.ax2.setLabel(text='Volume')

        self._market_activity = pg.PlotCurveItem(
            np.arrange(NUMBER_OF_TIMEPOINTS), np.arrange(NUMBER_OF_TIMEPOINTS),
            pen=pg.mkPen('k', style=Qt.DashLine, width=1)
        )
        self.p2.addItem(self._market_activity)

        # Automatically rescale our twinned Y axis



def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec_()

if __name__ == '__main__':
    main()