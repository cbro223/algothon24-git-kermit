
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd 
##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)

def getMyPositionOriginal(prcSoFar):
    # Provided code start 
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm
    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos
    # Provided Code end


def getMyPosition(prcSoFar):
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

