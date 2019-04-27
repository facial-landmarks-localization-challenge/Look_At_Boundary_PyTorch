import numpy as np
from sklearn.decomposition import PCA


def procrustes(X, Y, scaling=True, reflection='best'):
    n, m = X.shape
    ny, my = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()
    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    # scale to equal (unit) norm
    X0 = X0/normX if normX > 1e-6 else X0
    Y0 = Y0/normY if normY > 1e-6 else Y0
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)), 0)
    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    if reflection is not 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0
        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)
    traceTA = s.sum()
    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY if normY > 1e-6 else traceTA * normX
        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX
    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b*np.dot(muY, T)
    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform


# input as array
def pdb(dataset, allShapes, numBins):
    alignedShape = allShapes
    meanShape = np.mean(alignedShape, 1)
    for i in range(len(alignedShape[0])):
        _, tmpS, _ = procrustes(meanShape.reshape((-1, 2), order='F'),
                                alignedShape[:, i].reshape((-1, 2), order='F'))
        alignedShape[:, i] = tmpS.reshape((1, -1), order='F')
    
    meanShape = np.mean(alignedShape, 1)
    meanShape = meanShape.repeat(len(alignedShape[0])).reshape(-1, len(alignedShape[0]))
    alignedShape = alignedShape - meanShape
    pca = PCA(n_components=2) if dataset in ['AFLW', 'COFW'] else PCA(n_components=1)
    posePara = pca.fit_transform(np.transpose(alignedShape))
    
    absPosePara = np.abs(posePara[:, 1]) if dataset in ['AFLW', 'COFW'] else np.abs(posePara)
    maxPosePara = np.max(absPosePara)
    maxSampleInBins = np.max(np.histogram(absPosePara, numBins)[0])
    
    newIdx = np.array([])
    for i in range(numBins):
        tmp1 = set([index for index in range(len(absPosePara))
                    if absPosePara[index] >= i*maxPosePara/numBins])
        tmp2 = set([index for index in range(len(absPosePara))
                    if absPosePara[index] <= (i+1)*maxPosePara/numBins])
        tmpTrainIdx = np.array(list(tmp1 & tmp2))
        ratio = round(maxSampleInBins / len(tmpTrainIdx)) if len(tmpTrainIdx) > 0 else 0
        newIdx = np.insert(newIdx, 0, values=tmpTrainIdx.repeat(ratio), axis=0)
    return newIdx
