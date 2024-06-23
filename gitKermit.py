import numpy as np

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)
boughtStocks = np.zeros([nInst, 10])


def getMyPosition(prcSoFar):
    # Provided code start
    global currentPos
    (nins, nt) = prcSoFar.shape
    if nt < 2:
        return np.zeros(nins)

    lastRet = (prcSoFar[:, -1] / prcSoFar[:, -2]) - 1
    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm
    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
    currentPos = np.array([int(x) for x in currentPos + rpos])
    return currentPos


def getMyPositionV2(prcSoFar):
    global currentPos
    global boughtStocks
    (nins, nt) = prcSoFar.shape
    if nt < 2:
        return np.zeros(nins)

    period = 7
    periods = 14
    riskAjMom = np.zeros(nins)
    for i in range(nins):
        riskAjMom[i] = getStockRiskAdjMomentum(prcSoFar[i, :], period, periods)
    normRiskAdjMom = riskAjMom / np.sqrt(riskAjMom.dot(riskAjMom))
    # normRiskAdjMom = riskAjMom / sum(riskAjMom)
    changPos = np.array([x for x in 10000 * nins * normRiskAdjMom / period])
    changPos[changPos < 3000] = 0
    changPos = np.array([int(x) for x in changPos / prcSoFar[:, -1]])
    prevBought = np.zeros(nins)
    for i in range(nins):
        for j in range(int(period / 2)):
            if nt % (j + 1) == 0:
                prevBought[i] = boughtStocks[i, j]
                boughtStocks[i, j] = changPos[i]

    currentPos = np.array([x for x in currentPos + changPos - prevBought])
    return currentPos


def getStockRiskAdjMomentum(stockData, lenPeriod, numPeriods):
    """
    calculates the current risk adjusted moment for a given stock

    Inputs
    ------
    stockData: array
        all the stock data for the stock we want to calculate the risk adjusted moment for
    lenPeriod: int
        the length of period we are calculating for ie. week, month, year
    numPeriods: int
        the number of periods we will use to calculate the average momentum of the stock for

    Outputs
    -------
    riskAjMomentum: float
        the risk adjusted moment of the stock.

    """

    sumCurrentMomentum = np.zeros(numPeriods)

    for i in range(numPeriods):
        sumCurrentMomentum[i] += (stockData[-((i * lenPeriod) + 1)] / stockData[-(((i + 1) * lenPeriod) + 1)]) - 1

    # averaging
    avMomentum = sum(sumCurrentMomentum) / numPeriods

    # calculating the standard deviation
    sd = np.sqrt(sum((sumCurrentMomentum - avMomentum) ** 2) / (numPeriods - 1))

    # returning the risk adjusted momentum of the stock
    return avMomentum / sd

