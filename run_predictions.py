import os
import numpy as np
import json
import time
from PIL import Image

def compute_convolution(I, T, stride=1):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows, n_cols, n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''

    heatmap = -np.ones((n_rows, n_cols))
    for i in range(0, int(n_rows * 0.8) - T.shape[0], stride):
        for j in range(0, n_cols - T.shape[1], stride):
            window = I[i:i + T.shape[0], j:j + T.shape[1]]
            heatmap[i][j] = np.sum(window * T) / 255.

    '''
    END YOUR CODE
    '''

    return heatmap


def predict_boxes(heatmap, Tmap, Ts, pred_stride=30):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    pts = []
    for i in range(0, heatmap.shape[0] - pred_stride, pred_stride):
        for j in range(0, heatmap.shape[1] - pred_stride, pred_stride):
            local_max = np.unravel_index(np.argmax(heatmap[i:i+pred_stride, j:j+pred_stride],
                                                   axis=None), (pred_stride, pred_stride))
            pts.append((local_max[0] + i, local_max[1] + j))

    # Further suppression
    if pts:
        arr = np.array(pts)
        idxs = np.argsort(-heatmap[(arr[:, 0], arr[:, 1])])
        for i in idxs:
            pt = pts[i]
            far = True
            for tl_row, tl_col, br_row, br_col, score in output:
                if np.linalg.norm(abs(tl_row - pt[0]) + abs(tl_col - pt[1])) < pred_stride:
                    far = False
            if far:
                h, w, d = Ts[Tmap[pt]].shape
                output.append([int(pt[0]), int(pt[1]), int(pt[0] + h), int(pt[1] + w),
                               float(sigmoid(heatmap[pt]))])
            if len(output) >= 4:
                break

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I, Ts):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    all_heatmaps = np.zeros((I.shape[0], I.shape[1], len(Ts)))
    a = time.time()
    for i, T in enumerate(Ts):
        all_heatmaps[:, :, i] = compute_convolution(I, T) / T.size
    max_heatmap = np.amax(all_heatmaps, axis=2)
    Tmap = np.argmax(all_heatmaps, axis=2)
    output = predict_boxes(max_heatmap, Tmap, Ts)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = './hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

# Hand-selected templates
Ts = []
I = Image.open(os.path.join(data_path, 'RL-003.jpg'))
Ts.append(np.asarray(I)[203:209, 336:342])
I.close()
I = Image.open(os.path.join(data_path,'RL-011.jpg'))
Ts.append(np.asarray(I)[72:90, 355:373])
I.close()
I = Image.open(os.path.join(data_path,'RL-044.jpg'))
Ts.append(np.asarray(I)[284:297, 468:481])
I.close()
I = Image.open(os.path.join(data_path,'RL-116.jpg'))
Ts.append(np.asarray(I)[161:168, 359:366])
I.close()
I = Image.open(os.path.join(data_path,'RL-259.jpg'))
Ts.append(np.asarray(I)[223:229, 315:321])
I.close()
for i, T in enumerate(Ts):
    Tmean = np.mean(T)
    Ts[i] = (T.astype(np.float32) - Tmean) / Tmean

'''
Make predictions on the training set.
'''
start = time.time()
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I, Ts)

    if i % 10 == 0:
        print(i, time.time() - start)


# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I, Ts)

        if i % 10 == 0:
            print(i)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
