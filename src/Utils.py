import math

def Gaussian(x, y, sigma):
    return 0.5/(math.pi*sigma**2)*math.exp(-(x**2 + y**2)*0.5/sigma**2)