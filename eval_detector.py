import os
import json
import numpy as np
import matplotlib.pyplot as plt


def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    h = max(0, min(box_1[2], box_2[2]) - max(box_1[0], box_2[0]))
    w = max(0, min(box_1[3], box_2[3]) - max(box_1[1], box_2[1]))
    intersect = h * w
    union = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1]) + \
            (box_2[2] - box_2[0]) * (box_2[3] - box_2[1]) - intersect
    iou = intersect / union
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        preds_assigned = set()
        pred = list(filter(lambda tup: tup[4] >= conf_thr, pred))
        for i in range(len(gt)):
            max_iou = 0
            for j in range(len(pred)):
                if pred[j][4] >= conf_thr:
                    iou = compute_iou(pred[j][:4], gt[i])
                    if iou > max_iou:
                        best_j = j
                        max_iou = iou
            if max_iou > iou_thr:
                preds_assigned.add(best_j)
                TP += 1
            else:
                FN += 1
        FP += len(pred) - len(preds_assigned)

    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = './hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold.

# using (ascending) list of confidence scores as thresholds
confidence_thrs = []
for fname in preds_train:
    for tup in preds_train[fname]:
        confidence_thrs.append(tup[4])
confidence_thrs = np.sort(confidence_thrs)
for iou in [0.25, 0.5, 0.75]:
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou, conf_thr=conf_thr)
        #print(conf_thr, tp_train[i])

    # Plot training set PR curves
    p_train = tp_train / (tp_train + fp_train)
    r_train = tp_train / (tp_train + fn_train)
    plt.plot(r_train, p_train, label='iou_thr='+str(iou))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Training Set Performance')
plt.legend()
plt.savefig(os.path.join(preds_path, 'pr_train.png'))
plt.close()

if done_tweaking:
    print('Code for plotting test set PR curves.')
    for iou in [0.25, 0.5, 0.75]:
        tp_test = np.zeros(len(confidence_thrs))
        fp_test = np.zeros(len(confidence_thrs))
        fn_test = np.zeros(len(confidence_thrs))
        for i, conf_thr in enumerate(confidence_thrs):
            tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=iou,
                                                                   conf_thr=conf_thr)
            # print(conf_thr, tp_test[i])

        # Plot testing set PR curves
        p_test = tp_test / (tp_test + fp_test)
        r_test = tp_test / (tp_test + fn_test)
        plt.plot(r_test, p_test, label='iou_thr='+str(iou))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Test Set Performance')
    plt.legend()
    plt.savefig(os.path.join(preds_path, 'pr_test.png'))
    plt.close()