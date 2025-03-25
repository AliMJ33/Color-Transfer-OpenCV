import cv2
import numpy as np

def deviation(image):
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())
    return (lMean, lStd, aMean, aStd, bMean, bStd)

def color_transfer(source, target):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    source = source.astype("float32")
    target = target.astype("float32")

    (lMean_src, lStd_src, aMean_src, aStd_src, bMean_src, bStd_src) = deviation(source)
    (lMean_tar, lStd_tar, aMean_tar, aStd_tar, bMean_tar, bStd_tar) = deviation(target)

    (l, a, b) = cv2.split(target)
    l -= lMean_tar
    a -= aMean_tar
    b -= bMean_tar

    l = (lStd_tar / lStd_src) * l
    a = (aStd_tar / aStd_src) * a
    b = (bStd_tar / bStd_src) * b

    l += lMean_src
    a += aMean_src
    b += bMean_src

    l = np.clip(l, 0, 255)
    a = np.clip(l, 0, 255)
    b = np.clip(l, 0, 255)

    transfer = cv2.merge([l, a, b])
    transfer = transfer.astype("uint8")
    transfer = cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)
    return transfer


source = cv2.imread("nature11.jpg")
target = cv2.imread("2034636.jpg")
transferred = color_transfer(source, target)

cv2.imshow("Source", source)
cv2.imshow("Target", target)
cv2.imshow("Color Transfer", transferred)

cv2.waitKey(0)
cv2.destroyAllWindows()