
import numpy as np

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)
boughtStocks = np.zeros([nInst,10])

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
    currentPos = np.array([int(x) for x in currentPos+rpos])
    # Provided Code end
    # Convert to a pandas dataframe for and trasnpose for easier manipulation
    prcSoFar = pd.DataFrame(prcSoFar)
    prcSoFar = prcSoFar.T
    print(prcSoFar)
    #First linear model for basic prediction
    predicted_prices = fit_linear_regression_basic(prcSoFar)



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
    #normRiskAdjMom = riskAjMom / sum(riskAjMom)
    changPos = np.array([x for x in 10000 * nins * normRiskAdjMom / period])
    changPos[changPos < 3000] = 0
    changPos = np.array([int(x) for x in changPos / prcSoFar[:, -1]])
    prevBought = np.zeros(nins)
    for i in range(nins):
        for j in range(int(period/2)):
            if nt % (j+1) == 0:
                prevBought[i] = boughtStocks[i, j]
                boughtStocks[i, j] = changPos[i]

    currentPos = np.array([x for x in currentPos + changPos - prevBought])
    return currentPos

def getStockRiskAdjMomentum(stockData, lenPeriod, numPeriods):
    '''
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

    '''

    sumCurrentMomentum = np.zeros(numPeriods)
    """
    # extracting and reshaping the relevant stock data for which the momentum will be calculated for
    relevantData = stockData[(len(stockData) - (numPeriods * lenPeriod)):]
    relevantData.shape = (numPeriods, lenPeriod)
    """

    for i in range(numPeriods):
        sumCurrentMomentum[i] += (stockData[-((i * lenPeriod) + 1)] / stockData[-(((i + 1) * lenPeriod) + 1)]) - 1

    # averaging
    avMomentum = sum(sumCurrentMomentum) / numPeriods

    # calculating the standard deviation
    sd = np.sqrt(sum((sumCurrentMomentum - avMomentum) ** 2) / (numPeriods - 1))

    # returning the risk adjusted momentum of the stock
    return avMomentum / sd

def fit_linear_regression_basic(prcSoFar):
    """
    Fits a basic linear regression by just passing in price of last 14 days and returns a predicted price
    Parameters:
        prcSoFar: numpy array of shape (nInst, nt) containing the price of each stock at each time step
    Returns:
        numpy array of shape (nInst,) containing the predicted price of each stock
    """
    # Sets up the x arrays and predicted prices array
    numberOfDays = 14
    predicted_prices = np.zeros(50)
    # x starts at day 0, ends at day 13
    x = np.arange(0, numberOfDays).reshape(-1,1)
    # Go through every financial instrument and fit a linear regression model to it
    for i in range(prcSoFar.shape[1]):
        y = prcSoFar.iloc[-numberOfDays-1:-1,i]
        model = LinearRegression().fit(x, y)
        predicted_prices[i] = model.predict(np.array([numberOfDays]).reshape(-1,1))

    return pd.Series(predicted_prices)

