
from functools import reduce
import matplotlib.pyplot as plt
from dateutil import parser


def calculateEma(price, interval = 9, startEma = -1):
    #https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
    # (Closing price-EMA(previous day)) x multiplier + EMA(previous day)
    k = 2/(interval + 1)
    if startEma > 0:
        return reduce(lambda x,y: x + [ (y - x[-1]) * k + x[-1] ], price, [startEma])
    else:        
        subset = price[0:interval]
        sma = sum(subset) / len(subset)
        start = [sma] * interval    
        return reduce(lambda x,y: x + [ (y - x[-1]) * k + x[-1] ], price[interval:], start) 

def rewindEma(price, interval, startEma):
    k = 2/(interval + 1)   
    return reduce(lambda x,c: x + [ (-c*k + x[-1])  / (-k+1) ], price, [startEma])

def priceChangeToPrice(data, initial = 100):    
    return list(reduce(lambda x,y:  x + [ x[-1]+(x[-1]*y) ], data, list([initial]) ) )

def rewindPriceChangeToPrice(data, initial = 100):
    return list(reduce(lambda x,y:  x + [ x[-1] / (y+1.0) ], data, list([initial]) ) )


def debugPlot(data, debug, timeDomains = [1,5,15,30]):
    sample1Min = data[0:181]
    sample1Min = sample1Min[1::2]  # only want price
    
    trigger = debug["Trigger"]["parent"][0]
    trainingExampleId = debug["TrainingExample"]["id"]
    symbol = debug["TrainingExample"]["symbol"]["sym"]

    triggerData = [val for sublist in trigger["event"]["data"] for val in sublist]
    priceData = list(filter(lambda x: x["$type"] == "m.q.PriceTs", triggerData ))
    ema15Data = list(filter(lambda x: x["$type"] == "m.q.EmaTs" and x["data"]["timePeriod"] == 15, triggerData ))
    ema65Data = list(filter(lambda x: x["$type"] == "m.q.EmaTs" and x["data"]["timePeriod"] == 65, triggerData ))

    # we need to rewind these values through time now.
    rewindPrice1 = rewindPriceChangeToPrice(sample1Min[::-1], initial=priceData[0]["data"]["close"])
    
    #print("15: "+str(ema15Data[0]["data"]["ema"]))
    #print("65: "+str(ema65Data[0]["data"]["ema"]))
    rewindEma15 = rewindEma(rewindPrice1, 15, startEma = ema15Data[0]["data"]["ema"]) 
    rewindEma65 = rewindEma(rewindPrice1, 65, startEma = ema65Data[0]["data"]["ema"]) 
    #print("rewindPrice1: " + str(rewindPrice1[-1]))
    #print("rewindEma15: " + str(rewindEma15[-1]))
    #print("rewindEma65: " + str(rewindEma65[-1]))

    enterPrice = priceData[0]["data"]["close"]
    print("symbol: "+symbol)
    print("Training Example: " + trainingExampleId)
    print("enter price: " + str(enterPrice))
    print("enter time: " + priceData[0]["time"])
    time = parser.parse(priceData[0]["time"])
    print(time.minute)
    graph1 = priceChangeToPrice(sample1Min, initial=rewindPrice1[-1])
    ema15 = calculateEma(graph1, 15, startEma=graph1[0]) 
    ema65 = calculateEma(graph1, 65, startEma=graph1[0]) 
    series = [graph1, ema65, ema15]
    ind = 1
    
    for t in filter(lambda x: x != 1,timeDomains):
        start = (ind*180)+1
        end = ((ind+1)*180)+1
        sampleXMin = data[start:end]
        sampleXMin = sampleXMin[::2]
        remainder = (60+time.minute) % t
        #print("x: "+str(60+time.minute))
        #print("remainder: " + str(remainder) )
        #print(graph1[-(remainder+1)])

        rewindPriceX = rewindPriceChangeToPrice(sampleXMin[::-1], initial=graph1[-(remainder+1)])
        extra = 90*t - 90*timeDomains[ind-1]
        series = [([None] * extra) + x for x in series]

        graphX = priceChangeToPrice(sampleXMin, initial=rewindPriceX[-1])
        graphX = [[x]*t for x in graphX]
        graphX = [val for sublist in graphX for val in sublist][remainder:]
        series.append(graphX)
        ind = ind+1
    for x in series:
        plt.plot(x) 
    plt.show()  

   
    