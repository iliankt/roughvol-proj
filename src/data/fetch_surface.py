from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime
from threading import Thread
from threading import Event
import pandas as pd

class IBApp(EClient,EWrapper):

    def __init__(self):
        EClient.__init__(self,self)
        self.conId = None
        self.expirations = set()
        self.strikes = set()
        self.trading_classes = set()
        self.connected = Event()
        self.contract_done = Event()
        self.chain_done = Event()
        self.data = {}
        self.n_done = 0
        self.n_expected = None
        self.all_done = Event()

    def contractDetails(self, reqId, contractDetails):
        self.conId = contractDetails.contract.conId

    def contractDetailsEnd(self, reqId):
        self.contract_done.set()

    def securityDefinitionOptionParameter(self, reqId, exchange,
            underlyingConId, tradingClass, multiplier, expirations, strikes):
        self.expirations |= set(expirations)
        self.strikes     |= set(strikes)
        self.trading_classes.add(tradingClass)

    def securityDefinitionOptionParameterEnd(self, reqId):
        self.chain_done.set()

    def tickPrice(self, reqId, tickType, price, attrib):
        if price is None or price <= 0:
            return
        if tickType == 66:
            self.data[reqId]['bid'] = price
        elif tickType == 67:
            self.data[reqId]['ask'] = price
        elif tickType == 75:
            self.data[reqId]['close'] = price

    def tickSnapshotEnd(self, reqId):
        self.n_done += 1
        if self.n_expected is not None and self.n_done >= self.n_expected:
            self.all_done.set()


def run_loop(app):
    app.run()


def _start(app, clientId=1):
    app.connect('127.0.0.1', 7497, clientId)
    Thread(target=run_loop, args=(app,), daemon=True).start()
    app.connected.wait(timeout=5)


def _build_option(symbol, expiration, strike, right, tradingClass,
                  exchange='SMART', currency='USD', multiplier='100'):
    c = Contract()
    c.symbol = symbol
    c.secType = 'OPT'
    c.exchange = exchange
    c.currency = currency
    c.lastTradeDateOrContractMonth = expiration
    c.strike = strike
    c.right = right
    c.multiplier = multiplier
    c.tradingClass = tradingClass
    return c


def fetch_chain(symbol, secType='STK', exchange='SMART', currency='USD'):
    app = IBApp()
    _start(app)

    underlying = Contract()
    underlying.symbol = symbol
    underlying.secType = secType
    underlying.exchange = exchange
    underlying.currency = currency

    app.reqContractDetails(1, underlying)
    app.contract_done.wait(timeout=5)

    app.reqSecDefOptParams(2, symbol, "", secType, app.conId)
    app.chain_done.wait(timeout=10)
    app.disconnect()

    expirations = sorted(app.expirations)
    strikes = sorted(app.strikes)
    return expirations, strikes, app.trading_classes


def fetch_surface(symbol, expiration, strikes, tradingClass, rights=('C', 'P'),
                  exchange='SMART', currency='USD', multiplier='100'):
    app = IBApp()
    _start(app)
    app.reqMarketDataType(3)

    reqId = 0
    for strike in strikes:
        for right in rights:
            app.data[reqId] = {'maturity': expiration, 'strike': strike, 'right': right,
                               'bid': None, 'ask': None, 'close': None}
            option = _build_option(symbol, expiration, strike, right, tradingClass,
                                   exchange=exchange, currency=currency, multiplier=multiplier)
            app.reqMktData(reqId, option, "", True, False, [])
            reqId += 1

    app.n_expected = reqId
    app.all_done.wait(timeout=30)
    app.disconnect()

    return pd.DataFrame(list(app.data.values()))


if __name__ == '__main__':
    expirations, strikes, tclasses = fetch_chain('ESTX50', 'IND', 'EUREX', currency='EUR')
    if not expirations:
        raise RuntimeError("Chaine vide : verifie TWS et les parametres du sous-jacent")
    print("trading classes:", tclasses)
    expiration = expirations[2]
    atm = min(strikes, key=lambda k: abs(k - 5000))
    strikes_sel = [k for k in strikes if atm - 200 <= k <= atm + 200]
    tc = list(tclasses)[0]

    df = fetch_surface('SX5E', expiration, strikes_sel, tradingClass=tc,
                       exchange='EUREX', currency='EUR', multiplier='10')
    print(df)