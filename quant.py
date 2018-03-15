
from functools import reduce


def calculateEma(price, interval = 9, usePriceAsInitial = False):
    #https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
    k = 2/(interval + 1)
    if usePriceAsInitial:
        return reduce(lambda x,y: x + [ (y-x[-1]+1)*k + x[-1] ], price, [price[0]])
    else:        
        subset = price[0:interval]
        sma = sum(subset) / len(subset)
        start = [sma] * interval    
        return reduce(lambda x,y: x + [ (y-x[-1]+1)*k + x[-1] ], price[interval:], start)    
    