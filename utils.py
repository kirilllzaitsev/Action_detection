import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
from PIL import Image 

H, W = 300, 300

def plot_data_sample(dataset, idx):
    
    sample = dataset.__getitem__(idx)
    x,y, h, w = sample['bbox'].data
    fig, ax = plt.subplots(1)
    ax.imshow(sample['image'])
    box = patches.Rectangle((x,y),h,w, edgecolor='r', facecolor="none")
    ax.add_patch(box)
    plt.title(sample['bbox'])
    plt.show()
    
def plot_transformed_data_sample(dataset, idx, transform):

    sample = dataset.__getitem__(idx)
    transformed_image = sample['image']\
                        .permute(1,2,0)\
                        .clamp(0,1)

    x,y, h, w = sample['bbox'].data
    
    fig, ax = plt.subplots(1)
    ax.imshow(transformed_image)
    box = patches.Rectangle((x,y),
                            h, w, 
                            edgecolor='r', facecolor="none")
    ax.add_patch(box)
    plt.title("Transformed sample image")
    plt.show()