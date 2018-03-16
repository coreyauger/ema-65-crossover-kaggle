
from functools import reduce
import matplotlib.pyplot as plt


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


def debugPlot(data, debug, timeDomains = [1,5]):
    sample1Min = data[0:181]
    
    #print(sample90Min)
    print(sample1Min.shape)
    sample1Min = sample1Min[1::2]  # only want price
    
    #print(sample90Min.shape)
    trigger = debug["Trigger"]["parent"][0]
    trainingExampleId = debug["TrainingExample"]["id"]
    symbol = debug["TrainingExample"]["symbol"]["sym"]

    triggerData = [val for sublist in trigger["event"]["data"] for val in sublist]
    priceData = list(filter(lambda x: x["$type"] == "m.q.PriceTs", triggerData ))
    ema15Data = list(filter(lambda x: x["$type"] == "m.q.EmaTs" and x["data"]["timePeriod"] == 15, triggerData ))
    ema65Data = list(filter(lambda x: x["$type"] == "m.q.EmaTs" and x["data"]["timePeriod"] == 65, triggerData ))

    # we need to rewind these values through time now.
    rewindPrice1 = rewindPriceChangeToPrice(sample1Min[::-1], initial=priceData[0]["data"]["close"])
    
    print("15: "+str(ema15Data[0]["data"]["ema"]))
    print("65: "+str(ema65Data[0]["data"]["ema"]))
    rewindEma15 = rewindEma(rewindPrice1, 15, startEma = ema15Data[0]["data"]["ema"]) 
    rewindEma65 = rewindEma(rewindPrice1, 65, startEma = ema65Data[0]["data"]["ema"]) 
    print("rewindPrice1: " + str(rewindPrice1[-1]))
    print(rewindEma65)
    print("rewindEma15: " + str(rewindEma15[-1]))
    print("rewindEma65: " + str(rewindEma65[-1]))

    enterPrice = priceData[0]["data"]["close"]
    print("symbol: "+symbol)
    print("Training Example: " + trainingExampleId)
    print("enter price: " + str(enterPrice))
    print("enter time: " + priceData[0]["time"])
    graph1 = priceChangeToPrice(sample1Min, initial=rewindPrice1[-1])
    
    #print(graph)
    #print(graph[-1])
    ema15 = calculateEma(graph1, 15, startEma=graph1[0]) 
    ema65 = calculateEma(graph1, 65, startEma=graph1[0]) 
    series = [graph1, ema65, ema15]
    ind = 1
    
    for t in filter(lambda x: x != 1,timeDomains):
        start = (ind*180)+1
        end = ((ind+1)*180)+1
        sampleXMin = data[start:end]
        sampleXMin = sampleXMin[::2]
        rewindPriceX = rewindPriceChangeToPrice(sampleXMin[::-1], initial=priceData[0]["data"]["close"])
        extra = 90*t - 90*timeDomains[ind-1]-timeDomains[ind-1] 
        print("extra: "+ str(extra))
        series = [([graph1[0]] * extra) + x for x in series]

        graphX = priceChangeToPrice(sampleXMin, initial=rewindPriceX[-1])
        graphX = [[x]*t for x in graphX]
        graphX = [val for sublist in graphX for val in sublist]
        print(len(graphX))
        print(len(series[0]))
        series.append(graphX)
        ind = ind+1


    for x in series:
        plt.plot(x) 
    plt.show()  
    '''
    sample5Min = data[181:361]
    sample5Min = sample5Min[::2]
    rewindPrice5 = rewindPriceChangeToPrice(sample5Min[::-1], initial=priceData[0]["data"]["close"])
    graph5 = priceChangeToPrice(sample5Min, initial=rewindPrice5[-1])
    plt.plot(([graph1[0]] * 72*5) + graph1)
    plt.plot(([graph1[0]] * 72*5) + ema65)
    plt.plot(([graph1[0]] * 72*5) + ema15)
    graph5 = [[x]*5 for x in graph5]
    graph5 = [val for sublist in graph5 for val in sublist]
    plt.plot(graph5)
    plt.show()
    '''

   
    