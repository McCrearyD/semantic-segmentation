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


Args = namedtuple('Args', ['save_dir', 'arch', 'snapshot', 'dataset_cls'])

def setup_net(snapshot):
    """Quickly create a network for the given snapshot.
    
    Arguments:
        snapshot {string} -- Input snapshot, IE. kitti_best.pth
    
    Returns:
        [net] -- PyTorch model.
    """
    cudnn.benchmark = False
    torch.cuda.empty_cache()

    args = Args(
        save_dir='./save',
        arch='network.deepv3.DeepWV3Plus',
        snapshot=snapshot,
	dataset_cls=cityscapes)
    
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

    return net

def predict_image(net, img, frame=None):
    """Using the network generated from "setup_net(...)", make a prediction on the input image.
    
    Arguments:
        net {[pytorch model]} -- Ideally the network generated from "setup_net(...)".
        img {[numpy.array]} -- Input image.
    
    Keyword Arguments:
        frame {[type]} -- [description] (default: {None})
    
    Returns:
        [type] -- [description]
    """

    img_tensor = img_transform(img)
    img = img_tensor.unsqueeze(0).cuda()
    pred = net(img)

    pred = pred.cpu().numpy().squeeze()
    pred = np.argmax(pred, axis=0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    colorized = cityscapes.colorize_mask(pred)

    out_path = 'output_video_frame_%i.png' % frame if frame is not None else 'output_image.png'

    o = np.array(colorized.convert('RGB'))
    o = o[:, :, ::-1].copy()
    return o

def predict_video(net, input_path, output_path, verbose=True, every_nth_frame=None):
    """Using the network generated from "setup_net(...)", make a prediction on the input video.
    
    Arguments:
        net {pytorch model} -- Ideally the network generated from "setup_net(...)".
        input_path {type} -- Path to input video.
        output_path {type} -- Path to output video.
    
    Keyword Arguments:
        verbose {bool} -- Print extra stuff. (default: {True})
        every_nth_frame {int} -- If not None, will only do inferences on every Nth frame. (default: {None})
    """

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(input_path)
    out_video = cv2.VideoWriter(input_path, fourcc, 30.0, (int(cap.get(3)),int(cap.get(4))))

    # predict
    with torch.no_grad():
        i = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                i += 1
                
                if every_nth_frame is not None and i % every_nth_frame != 0:
                    continue

                # predict & write to output buffer
                seg_frame = predict_image(frame, i)
                out_video.write(cv2.cvtColor(seg_frame, cv2.COLOR_RGB2BGR))
            else:
                break
            if verbose:
                print('Frame %i' % i)
    if verbose:
        print('Finished.')
    cap.release()
    out_video.release()
