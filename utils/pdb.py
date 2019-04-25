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


if __name__ == '__main__':
    import cv2
    from utils import get_annotations_list, kp_num, \
        dataset_route, crop_size, cropped_pic_kp
    numBin = 17
    use_dataset, use_split = 'AFLW', 'train'
    annotations = get_annotations_list(use_dataset, use_split)
    kp_num, length = kp_num[use_dataset], len(annotations)
    allShape = np.zeros((2*kp_num, length))
    for line_index, line in enumerate(annotations):
        # pic = cv2.imread(dataset_route[use_dataset] + line[-1])
        coord_x, coord_y = [], []
        for kp_index in range(kp_num):
            coord_x.append(float(line[2 * kp_index]))
            coord_y.append(float(line[2 * kp_index + 1]))
        position_before = np.float32([[int(line[-5]), int(line[-3])],
                                      [int(line[-5]), int(line[-2])],
                                      [int(line[-4]), int(line[-2])]])
        position_after = np.float32([[0, 0],
                                     [0, crop_size - 1],
                                     [crop_size - 1, crop_size - 1]])
        crop_matrix = cv2.getAffineTransform(position_before, position_after)
        # pic_crop = cv2.warpAffine(pic, crop_matrix, (crop_size, crop_size))
        coord_x_after, coord_y_after = cropped_pic_kp(use_dataset, crop_matrix, 
                                                      coord_x, coord_y)
        for data_index in range(kp_num):
            allShape[data_index][line_index] = float(coord_x_after[data_index])
            allShape[data_index+kp_num][line_index] = float(coord_y_after[data_index])
        
        # import cv2
        # for coord_index in range(kp_num):
        #     cv2.circle(
        #                pic_crop, 
        #                (int(allShape[coord_index][line_index]),
        #                 int(allShape[coord_index+kp_num][line_index])),
        #                1,
        #                (0,0,255)
        #               )
        # cv2.imshow('pic', pic_crop)
        # cv2.moveWindow('pic', 0, 0)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    newIdxs = pdb(use_dataset, allShape, numBin)
    print(len(newIdxs))

    # for id_index in newIdxs:
    #     line = annotations[int(id_index)]
    #     pic = cv2.imread(dataset_route[use_dataset] + line[-1])
    #     position_before = np.float32([[int(line[-5]), int(line[-3])],
    #                                   [int(line[-5]), int(line[-2])],
    #                                   [int(line[-4]), int(line[-2])]])
    #     position_after = np.float32([[0, 0],
    #                                  [0, crop_size - 1],
    #                                  [crop_size - 1, crop_size - 1]])
    #     crop_matrix = cv2.getAffineTransform(position_before, position_after)
    #     pic_crop = cv2.warpAffine(pic, crop_matrix, (crop_size, crop_size))
    #     cv2.imshow('pic', pic_crop)
    #     cv2.moveWindow('pic', 0, 0)
    #     cv2.waitKey(500)
    #     cv2.destroyAllWindows()
