import os
import sys
import argparse
from PIL import Image
import numpy as np
import cv2

import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg
from collections import namedtuple
from time import time


class Args(object):
    def __init__(self, save_dir, arch, snapshot):
        self.save_dir = save_dir
        self.arch = arch
        self.snapshot = snapshot
        self.dataset_cls = cityscapes

# Args = namedtuple('Args', ['save_dir', 'arch', 'snapshot', 'dataset_cls'])

def setup_net(snapshot):
    """Quickly create a network for the given snapshot.
    
    Arguments:
        snapshot {string} -- Input snapshot, IE. kitti_best.pth
    
    Returns:
        [net, transform] -- PyTorch model & the image transform function.
    """
    cudnn.benchmark = False
    torch.cuda.empty_cache()

    args = Args('./save', 'network.deepv3.DeepWV3Plus', snapshot)
    
    assert_and_infer_cfg(args, train_mode=False)
    # get net
    net = network.get_net(args, criterion=None)
    net = torch.nn.DataParallel(net).cuda()
    print('Net built.')

    net, _ = restore_snapshot(net, optimizer=None, snapshot=snapshot, restore_optimizer_bool=False)
    net.eval()
    print('Net restored.')

    # get data
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

    return net, img_transform, args

def predict_image(net, img_transform, args, img):
    """Using the network generated from "setup_net(...)", make a prediction on the input image.
    
    Arguments:
        net {pytorch model} -- Ideally the network generated from "setup_net(...)".
        img_transform {transform} -- Output from setup_net.
        args {Args} -- Arguments for the network from setup_net.
        img {numpy.array} -- Input image.
    
    Keyword Arguments:
        frame {type} -- [description] (default: {None})
    
    Returns:
        [np.array, np.array] -- Colorized & non-colorized predictions resepectively.
    """

    img_tensor = img_transform(img)

    with torch.no_grad():
	    img = img_tensor.unsqueeze(0).cuda()
	    pred = net(img)

    pred = pred.cpu().numpy().squeeze()
    pred = np.argmax(pred, axis=0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    colorized = cityscapes.colorize_mask(pred)

    o = np.array(colorized.convert('RGB'))
    o = o[:, :, ::-1].copy()
    return o, pred

def predict_video(net, img_transform, args, input_path, output_path, verbose=True, every_nth_frame=None):
    """Using the network generated from "setup_net(...)", make a prediction on the input video.
    
    Arguments:
        net {pytorch model} -- Ideally the network generated from "setup_net(...)".
        img_transform {transform} -- Output from setup_net.
        args {Args} -- Arguments for the network from setup_net.
        input_path {type} -- Path to input video.
        output_path {type} -- Path to output video.
    
    Keyword Arguments:
        verbose {bool} -- Print extra stuff. (default: {True})
        every_nth_frame {int} -- If not None, will only do inferences on every Nth frame. (default: {None})
    """

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(input_path)

    if output_path.endswith('.mp4'):
        output_path = output_path[:-4] # shave off end

    # create 2 buffers for our mask & overlay output videos
    out_video_mask = cv2.VideoWriter(output_path + '_mask.mp4', fourcc, 30.0, (int(cap.get(3)),int(cap.get(4))))
    out_video_mask_overlay = cv2.VideoWriter(output_path + '_overlaid.mp4', fourcc, 30.0, (int(cap.get(3)),int(cap.get(4))))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_process_frames = (frame_count / (1 if every_nth_frame is None else every_nth_frame))

    if verbose:
        print('Running inference on video with path: %s' % input_path)
        print('Frames in video: %i ' % (frame_count))
        print('Total frames to be processed: %i ' % total_process_frames)
        print()

    i = 0
    c = 0
    start = time()
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        i += 1
        
        if every_nth_frame is not None and i % every_nth_frame != 0:
            continue

        c += 1

        # predict & write to output buffer
        seg_frame_colorized, seg_frame_gray = predict_image(net, img_transform, args, frame)
        seg_frame_colorized = cv2.cvtColor(seg_frame_colorized, cv2.COLOR_RGB2BGR)

        background = Image.fromarray(frame)
        foreground = Image.fromarray(seg_frame_colorized)
        foreground.putalpha(128)
        background.paste(foreground, (0, 0), foreground)
        background = cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR)

        # write to both output buffers
        out_video_mask.write(seg_frame_colorized) # just the mask
        out_video_mask_overlay.write(background) # mask with the overlay

        if verbose:
            if i > 1:
                ret = '\r'
            else:
                ret = ''
            average = (time() - start) / c
            frames_left = total_process_frames - c
            s = 'Frame %i/%i' % (i, frame_count)
            s += '\tAverage Inference: %.3fs' % average
            s += '\tETA: %.1fm' % (frames_left * average / 60)
            print(s, end=ret)
    if verbose:
        print('Finished.', end='\r')
        print()
        
    cap.release()
    out_video_mask.release()
    out_video_mask_overlay.release()
