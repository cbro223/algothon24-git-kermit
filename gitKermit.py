import numpy as np

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)
boughtStocks = np.zeros([nInst, 10])
firstRun = True
table = 0


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
    # Provided Code end


def getMyPositionLinearRegression(prcSoFar):
    global currentPos
    # Convert to a pandas dataframe for and trasnpose for easier manipulation
    prcSoFar = pd.DataFrame(prcSoFar)
    prcSoFar = prcSoFar.T
    #First linear model for basic prediction
    predicted_prices = fit_linear_regression_basic(prcSoFar)

    prcSoFar.loc["Last-day-prediction"] = predicted_prices
    
    # This model is dogshit, would not recommend using it lol 
    lastRet = np.log(prcSoFar.iloc[-1, :] / prcSoFar.iloc[-2, :])
    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm
    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar.iloc[-1,:]])
    currentPos = np.array([int(x) for x in currentPos+rpos])

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


def getMyPositionV3(prcSoFar):
    global currentPos
    global boughtStocks
    global firstRun
    global table
    (nins, nt) = prcSoFar.shape
    if nt < 2:
        return np.zeros(nins)
    if firstRun:
        table = CreateLeadLagProbabilityTable(prcSoFar, 14, 7)
        firstRun = False
    else:
        table = UpdateLeadLagProbabilityTable(table , prcSoFar, 14, 7)

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

    x = np.arange(0, numberOfDays + 1).reshape(-1,1)
    # Go through every financial instrument and fit a linear regression model to it 

    for i in range(prcSoFar.shape[1]):
        y = prcSoFar.iloc[-numberOfDays-1:,i]
        model = LinearRegression().fit(x, y)
        predicted_prices[i] = model.predict(np.array([numberOfDays]).reshape(-1,1))

    return pd.Series(predicted_prices)

def UpdateLeadLagProbabilityTable(table, stockData, maxLeadingTime, maxPeriodOfLeading):

    for overNumDays in range(1,maxPeriodOfLeading+1):
        percentChangeInPriceData = (stockData[:,overNumDays:]/stockData[:,:-overNumDays]) - 1
        for leadingStock in range(np.shape(percentChangeInPriceData)[0]):
            for daysLeading in range(1,maxLeadingTime + 1):
                relativeChangeInPrice = percentChangeInPriceData[leadingStock,-(daysLeading + 1)]/percentChangeInPriceData[:, -1] - 1
                index1 = relativeChangeInPrice <= 0.7
                index2 = relativeChangeInPrice >= -0.7
                relativeChangeInPrice[index1 == index2] = 1
                relativeChangeInPrice[index1 != index2] = 0
                table[leadingStock,:,daysLeading,overNumDays] += relativeChangeInPrice

    return table

def CreateLeadLagProbabilityTable(stockData, maxLeadingTime, maxPeriodOfLeading):
    """
    creates a 4 dimensional table which contins the frequency of time that a particular stock exhibits the same
    behaviour as another stock after some time i.e. a tabel that show the number of times that stock a leads stock b
    by a certain amount of time over a certain period. this tabel can be indexed such that table[i,j,k,l] shows the
    number of times that stock i leads stock j by k days in regard to the price change over l days

    Inputs
    ------
    stockData: numpy array
        the complete stock data so far
    maxLeadingTime: int
        the maximum amount of days of leading that you want to calculate i.e., up to stock 1 leads stock 2 by
        maxLeadingTime day.
    maxPeriodOfLeading: int
        the maximum amount of days over which the change in price is calculated. i.e. up to stock 1 leads stock 2 by k
        day in regard to the price change of the stocks over maxPeriodOfLeading days

    Outputs
    -------
    table : numpy array
        a frequency of time that stock i leads stock j by k days in regard to the price change of the stock over l days

    Note
    1) the change in price is calculated in reliteve chang in price not absolute change in price
    """
    table = np.zeros([stockData.shape[0],stockData.shape[0], maxLeadingTime + 1, maxPeriodOfLeading + 1])
    for overNumDays in range(1, maxPeriodOfLeading + 1):
        percentChangeInPriceData = (stockData[:, overNumDays:] / stockData[:, :-overNumDays]) - 1
        for leadingStock in range(np.shape(percentChangeInPriceData)[0]):
            for daysLeading in range(0,maxLeadingTime + 1):
                if daysLeading == 0:
                    relativeChangeInPrice = percentChangeInPriceData[leadingStock, :] / percentChangeInPriceData[:, :] - 1
                else:
                    relativeChangeInPrice = percentChangeInPriceData[leadingStock, :-daysLeading] / percentChangeInPriceData[:,daysLeading:] - 1
                index1 = relativeChangeInPrice <= 0.7
                index2 = relativeChangeInPrice >= -0.7
                relativeChangeInPrice[index1 == index2] = 1
                relativeChangeInPrice[index1 != index2] = 0
                relativeChangeInPrice = np.sum(relativeChangeInPrice, axis=1)
                table[leadingStock, :, daysLeading, overNumDays] += relativeChangeInPrice
    return table


def UpdateLeadLagProbabilityTable(table, stockData, maxLeadingTime, maxPeriodOfLeading):

    for overNumDays in range(1,maxPeriodOfLeading+1):
        percentChangeInPriceData = (stockData[:,overNumDays:]/stockData[:,:-overNumDays]) - 1
        for leadingStock in range(np.shape(percentChangeInPriceData)[0]):
            for daysLeading in range(1,maxLeadingTime + 1):
                relativeChangeInPrice = percentChangeInPriceData[leadingStock,-(daysLeading + 1)]/percentChangeInPriceData[:, -1] - 1
                index1 = relativeChangeInPrice <= 0.7
                index2 = relativeChangeInPrice >= -0.7
                relativeChangeInPrice[index1 == index2] = 1
                relativeChangeInPrice[index1 != index2] = 0
                table[leadingStock,:,daysLeading,overNumDays] += relativeChangeInPrice

    return table


def CreateLeadLagProbabilityTable(stockData, maxLeadingTime, maxPeriodOfLeading):
    """
    Creates a four dimensional table which contins the frequency of time that a particular stock exhibits the same
    behaviour as another stock after some time i.e., a tabel that show the number of times that stock a leads stock b
    by a certain amount of time over a certain period. This tabel can be indexed such that table[i, j, k, l] shows the
    number of times that stock i leads stock j by k days in regard to the price change over l days

    Inputs
    ------
    stockData: numpy array
        the complete stock data so far
    maxLeadingTime: int
        the maximum number of days of leading that you want to calculate i.e., up to stock 1 leads stock 2 by
        maxLeadingTime day.
    maxPeriodOfLeading: int
        the maximum number of days over which the change in price is calculated. I.e. up to stock 1 leads stock 2 by k
        day in regard to the price change of the stocks over maxPeriodOfLeading days

    Outputs
    -------
    table : numpy array
        a frequency of time that stock i leads stock j by k days in regard to the price change of the stock over l days

    Note
    1) the change in price is calculated in relative change in price, not absolute change in price
    """
    table = np.zeros([stockData.shape[0],stockData.shape[0], maxLeadingTime + 1, maxPeriodOfLeading + 1])
    for overNumDays in range(1, maxPeriodOfLeading + 1):
        percentChangeInPriceData = (stockData[:, overNumDays:] / stockData[:, :-overNumDays]) - 1
        for leadingStock in range(np.shape(percentChangeInPriceData)[0]):
            for daysLeading in range(0,maxLeadingTime + 1):
                if daysLeading == 0:
                    relativeChangeInPrice = percentChangeInPriceData[leadingStock, :] / percentChangeInPriceData[:, :] - 1
                else:
                    relativeChangeInPrice = percentChangeInPriceData[leadingStock, :-daysLeading] / percentChangeInPriceData[:,daysLeading:] - 1
                index1 = relativeChangeInPrice <= 0.7
                index2 = relativeChangeInPrice >= -0.7
                relativeChangeInPrice[index1 == index2] = 1
                relativeChangeInPrice[index1 != index2] = 0
                relativeChangeInPrice = np.sum(relativeChangeInPrice, axis=1)
                table[leadingStock, :, daysLeading, overNumDays] += relativeChangeInPrice
    return table

