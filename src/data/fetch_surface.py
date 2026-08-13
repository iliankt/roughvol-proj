from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime, timezone
from threading import Thread
from threading import Event
import pandas as pd
import os
import glob


class IBApp(EClient, EWrapper):

    def __init__(self):
        EClient.__init__(self, self)
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
    _start(app, clientId=1)

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
    _start(app, clientId=2)
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

def save_surface(df, symbol, spot=None, base_dir='data', delayed=True):
    expiration = str(df['maturity'].iloc[0])
    ts = datetime.now(timezone.utc)
    ts_str = ts.strftime('%Y%m%d_%H%M%S')

    df = df.copy()
    df['symbol'] = symbol
    df['snapshot_utc'] = ts.isoformat()
    df['spot'] = spot
    df['delayed'] = delayed

    out_dir = os.path.join(base_dir, symbol)
    os.makedirs(out_dir, exist_ok=True)

    path = path = os.path.join(out_dir, f'surface_{symbol}_{expiration}.parquet')
    df.to_parquet(path, index=False)
    print(f"Snapshot sauvegarde : {path}  ({len(df)} lignes)")
    return path


def load_latest_surface(symbol, base_dir='data'):

    pattern = os.path.join(base_dir, symbol, f'surface_{symbol}_*.parquet')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Aucun snapshot pour {symbol} dans {base_dir}")
    latest = files[-1]
    print(f"Chargement : {latest}")
    return pd.read_parquet(latest)


if __name__ == '__main__':
    spot = 304.76

    expirations, strikes, tclasses = fetch_chain('AAPL')
    if not expirations:
        raise RuntimeError("Chaine vide : verifie TWS")

    exp_choisies = [expirations[9], expirations[13]]

    atm = min(strikes, key=lambda k: abs(k - spot))
    strikes_sel = [k for k in strikes if atm - 50 <= k <= atm + 50]

    for i, expiration in enumerate(exp_choisies, start=1):
        print(f"\n=== Maturite T{i} : {expiration} ===")
        df = fetch_surface('AAPL', expiration, strikes_sel, tradingClass='AAPL',
                           exchange='SMART', currency='USD', multiplier='100')
        print(df)
        save_surface(df, 'AAPL', spot=spot)