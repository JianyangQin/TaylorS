import pywt

def disentangle(x, w, j):
    x = x.transpose(0,2,1,3) # [S,D,N,T]
    coef = pywt.wavedec(x, w, level=j)
    coefl = [coef[0]]
    for i in range(len(coef)-1):
        coefl.append(None)
    coefh = [None]
    for i in range(len(coef)-1):
        coefh.append(coef[i+1])
    xl = pywt.waverec(coefl, w).transpose(0,2,1,3)
    xh = pywt.waverec(coefh, w).transpose(0,2,1,3)
    return xl, xh