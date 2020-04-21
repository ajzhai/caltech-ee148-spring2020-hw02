import os
import json
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

def plot_bbox(img, coords, gt_coords=None):
    fig, ax = plt.subplots(1)
    fig.set_size_inches(8, 6)
    I = Image.open(os.path.join(data_path, img))
    ax.set_title(img)
    ax.imshow(I)
    for coord in coords:
        tli, tlj, bri, brj, conf = coord
        if conf > 0.54:
            x, y = tlj - 1, tli - 1
            w, h = brj - tlj + 1, bri - tli + 1
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)
            ax.axis('off')
    if gt_coords:
        for coord in gt_coords:
            tli, tlj, bri, brj = coord
            x, y = tlj - 1, tli - 1
            w, h = brj - tlj + 1, bri - tli + 1
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='orange', facecolor='none')
            ax.add_patch(rect)
            ax.axis('off')

    plt.savefig(os.path.join(preds_path, img[:-4] + '_out.png'))
    plt.show()
    plt.close()


data_path = '../data/RedLights2011_Medium'
preds_path = './hw02_preds'

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
fig, axs = plt.subplots(1, len(Ts))
fig.set_size_inches(12, 4)
for i in range(len(Ts)):
    axs[i].imshow(Ts[i])
plt.savefig('templates_plot.png')
plt.close()

gts_path = '../data/hw02_annotations'
with open(os.path.join(gts_path, 'formatted_annotations_students.json'), 'r') as f:
    gts = json.load(f)

with open(os.path.join(preds_path, 'preds_test.json'), 'r') as f:
    preds = json.load(f)
test_imgs = ['RL-092.jpg', 'RL-115.jpg', 'RL-271.jpg', 'RL-326.jpg',
             'RL-030.jpg', 'RL-140.jpg', 'RL-151.jpg', 'RL-167.jpg']
for im in test_imgs:
    plot_bbox(im, preds[im], gts[im])
