from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from datetime import datetime
from threading import Thread
from threading import Event

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
    