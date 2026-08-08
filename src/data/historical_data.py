import numpy as np
import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime
from threading import Thread,Event
from utils import write_date,plan_backfill
from datetime import timezone
import time

class IBApp(EClient,EWrapper):

    def __init__(self):
        EClient.__init__(self,self)
        self.data = []
        self.done = Event()
        self.error_code = None
        self.connected = Event()

    def historicalData(self, reqId, bar):
        self.data.append({'time' : bar.time,
                          'open' : bar.open,
                          'close' : bar.close,
                          'high' : bar.high,
                          'low' : bar.low})

    def historicalDataEnd(self, reqId, start, end):
        self.done.set()

    def error(self, reqId, errorCode, errorString,advancedOrderRejectJson=""):
            goodErrors = [2104, 2106, 2158, 2176]
            badErrors = [162, 165, 200, 354, 10314]
            if errorCode in goodErrors:
                return
            
            elif errorCode in badErrors:
                print(errorString)
                self.error_code = errorCode
                self.done.set()
    
            else:
                print(errorString)
    
    def nextValidId(self, orderId):
        self.connected.set()

def run(app):
    app.run()

def get_historical_data(symbol,bar_size,what_to_show,end_time,duration_str):
    app = IBApp()
    app.connect()
    app_threading = Thread(target=run,args=(app,))
    app_threading.start()
    app.connected.wait(timeout=5)

    contract = Contract()
    contract.symbol = symbol
    contract.secType = 'STK'
    contract.exchange = 'SMART'
    contract.currency = 'USD'

    app.reqHistoricalData(reqId=1,
                          contract=contract,
                          endDateTime=end_time,
                          durationStr=duration_str,
                          barSizeSetting=bar_size,
                          whatToShow=what_to_show,
                          useRTH=1,
                          formatDate=2,
                          keepUpToDate=0,
                          chartOptions=[])

    check = app.done.wait(timeout=30)
    app.disconnect()

    if check and app.error_code is None:
            df = pd.DataFrame(app.data)
            df['datetime'] = pd.to_datetime(df['datetime'].astype('int64'),unit='s',utc=True).astype('datetime64[ns, UTC]')
            df['symbol'] = symbol
            df['whatToShow'] = what_to_show
            df['barSize'] = bar_size
            df = df.rename(columns={'datetime': 'ts_utc'})
            df = df[['ts_utc','open','high','low','close','volume','wap','barCount','symbol','barSize','whatToShow']]
            for jour, df_jour in df.groupby(df['ts_utc'].dt.date):   
                write_date(df_jour,symbol,bar_size,what_to_show)

if __name__ == '__main__':
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tranches = plan_backfill(start, end,'5 mins')
    for end_time, duration_str in tranches:
            get_historical_data('AAPL', '5 mins', 'TRADES', end_time, duration_str)
            time.sleep(1)