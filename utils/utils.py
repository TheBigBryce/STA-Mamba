
import torch
import torch.nn as nn
import numpy as np
from medpy import metric
from scipy.ndimage import zoom
import seaborn as sns
from PIL import Image 
import matplotlib.pyplot as plt
import imageio
import pdb
import matplotlib
matplotlib.use('agg')

from segmentation_mask_overlay import overlay_masks
import matplotlib.colors as mcolors

from matplotlib.patches import Patch
from matplotlib.colors import to_rgb

import SimpleITK as sitk
import pandas as pd

import cv2
import time

from thop import profile
from thop import clever_format

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
    
def one_hot_encoder(input_tensor,dataset,n_classes = None):
    tensor_list = []
    if dataset == 'MMWHS':  
        dict = [0,205,420,500,550,600,820,850]
        for i in dict:
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    else:
        for i in range(n_classes):
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()    

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        #print(inputs)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        asd = metric.binary.assd(pred, gt)
        return dice, hd95, jaccard, asd
        # return dice,0,0,0
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0


def calculate_metric_percase_dice(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        #hd95 = metric.binary.hd95(pred, gt)
        #jaccard = metric.binary.jc(pred, gt)
        #asd = metric.binary.assd(pred, gt)
        return dice, 0.0, 0.0, 0.0
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0




def calculate_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    else:
        return 0



def test_single_volume_dice(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, class_names=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if class_names==None:
        mask_labels = np.arange(1,classes)
    else:
        mask_labels = class_names
    cmaps = mcolors.CSS4_COLORS
    my_colors=['red','darkorange','yellow','forestgreen','blue','purple','magenta','cyan','deeppink', 'chocolate', 'olive','deepskyblue','darkviolet']
    cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:classes-1]}

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                P = net(input)
                outputs = 0.0
                for idx in range(len(P)):
                    outputs += P[idx]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
                
                lbl = label[ind, :, :]
                masks = []
                for i in range(1, classes):
                    masks.append(lbl==i)
                preds_o = []
                for i in range(1, classes):
                    preds_o.append(pred==i)
                if test_save_path is not None:
                    fig_gt = overlay_masks(image[ind, :, :], masks, labels=mask_labels, colors=cmap, mask_alpha=0.5)
                    fig_pred = overlay_masks(image[ind, :, :], preds_o, labels=mask_labels, colors=cmap, mask_alpha=0.5)
                    # Do with that image whatever you want to do.
                    fig_gt.savefig(test_save_path + '/'+case + '_' +str(ind)+'_gt.png', bbox_inches="tight", dpi=300)
                    fig_pred.savefig(test_save_path + '/'+case + '_' +str(ind)+'_pred.png', bbox_inches="tight", dpi=300)
                    plt.close('all')

    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            P = net(input)
            outputs = 0.0
            for idx in range(len(P)):
                outputs += P[idx]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase_dice(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list






def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, class_names=None):
    # Prepare input data
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    # pdb.set_trace()

    # Define mask labels and colors
    mask_labels = np.arange(1, classes) if class_names is None else class_names
    my_colors = ['red', 'darkorange', 'yellow', 'forestgreen', 'blue', 'purple', 
                 'magenta', 'cyan', 'deeppink', 'chocolate', 'olive', 'deepskyblue', 'darkviolet']
    cmaps = mcolors.CSS4_COLORS
    cmap = {color: cmaps[color] for color in my_colors if color in cmaps}

    # Initialize prediction volume
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)

        for ind in range(image.shape[0]):

        # for ind in range(102, 103):

            if np.sum(label[ind,:,:]) == 0:

                continue

            # ind = 105
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]

            # Resize slice to match patch size if needed
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)

            input_tensor = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()

            # Model inference
            net.eval()
            with torch.no_grad():
                t1 = time.time()
                outputs = net(input_tensor)
                print("Inference time: ", time.time()-t1)

                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().detach().numpy()

            # Resize prediction back to original size if resized earlier
            if x != patch_size[0] or y != patch_size[1]:
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                pred = out

            prediction[ind] = pred

            # Extract ground truth and predictions for each class
            lbl = label[ind, :, :]
            masks = [(lbl == i) for i in range(1, classes)]
            preds_o = [(pred == i) for i in range(1, classes)]

            # Save test results if a path is provided
            if test_save_path is not None:
                # image_slice = (image[ind, :, :] * 255).astype(np.uint8)
                # image_slice_expanded = np.expand_dims(image_slice, axis=0)
                # repeated_images = np.repeat(image_slice_expanded, 8, axis=0)

                # # sanity check
                # random_bool_matrix = np.random.choice([True, False], size=(8, 512, 512))

                # cv2.imwrite(f"{test_save_path}/{case}_{ind}_img.png", repeated_images[0])

                # # Overlay masks for ground truth and predictions
                # fig_gt = custom_overlay_masks(repeated_images, masks, labels=mask_labels, colors=cmap, alpha=0.5)
                # fig_gt.savefig(f"{test_save_path}/{case}_{ind}_gt.png", bbox_inches="tight", dpi=300)

                # fig_pred = custom_overlay_masks(repeated_images, preds_o, labels=mask_labels, colors=cmap, alpha=0.5)
                # fig_pred.savefig(f"{test_save_path}/{case}_{ind}_pred.png", bbox_inches="tight", dpi=300)

                # plt.close('all')  # Close plots to free memory

                print(f"Saved image - {test_save_path}/{case}_{ind}")


    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            P = net(input)
            outputs = 0.0
            for idx in range(len(P)):
                outputs += P[idx]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
        # metric = calculate_metric_percase(prediction[102,:,:], label[102,:,:])

    if test_save_path is not None:
        print('here')
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def custom_overlay_masks(image, masks, labels=None, colors=None, alpha=0.5):
    """
    Overlay masks and predictions on the image.
    
    Args:
        image (numpy.ndarray): The original image (2D or 3D).
        masks (numpy.ndarray): Binary masks of shape [num_classes, height, width].
        labels (list or np.ndarray): Class labels corresponding to masks.
        colors (dict): A dictionary mapping labels to colors (optional).
        alpha (float): Transparency level for overlaying masks.
    
    Returns:
        matplotlib.figure.Figure: The figure with overlays.
    """
    # pdb.set_trace()
    if labels is None:
        labels = np.arange(1, masks.shape[0] + 1)
    
    # Default colors
    my_colors = ['red', 'darkorange', 'yellow', 'forestgreen', 'blue', 'purple', 'magenta', 
                 'cyan', 'deeppink', 'chocolate', 'olive', 'deepskyblue', 'darkviolet']
    # if colors is None:
    colors = {label: my_colors[i % len(my_colors)] for i, label in enumerate(labels)}
    
    # Prepare the base image
    if len(image.shape) == 3:  # If image has multiple channels
        base_image = np.mean(image, axis=0)
    else:
        base_image = image
    base_image = base_image / np.max(base_image)  # Normalize to [0, 1]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(base_image, cmap='gray')

    # Overlay masks
    for i, label in enumerate(labels):
        mask = masks[i]
        color = colors.get(label, "red")
        overlay = np.zeros((*mask.shape, 4))  # RGBA
        overlay[..., :3] = to_rgb(color)  # Convert color to RGB
        overlay[..., 3] = mask * alpha  # Alpha channel based on the mask
        ax.imshow(overlay, interpolation='none')

    # Add legend
    try:
        patches = [Patch(color=to_rgb(colors[label]), label=f"Class {label}") for label in labels]
    except KeyError as e:
        print(f"KeyError: {e} - Label not found in colors dictionary.")
        return fig
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    ax.axis('off')
    plt.tight_layout()
    return fig

def val_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():                
                P = net(input)
                outputs = 0.0
                for idx in range(len(P)):
                   outputs += P[idx]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            P = net(input)
            outputs = 0.0
            for idx in range(len(P)):
               outputs += P[idx]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_dice_percase(prediction == i, label == i))
    return metric_list

