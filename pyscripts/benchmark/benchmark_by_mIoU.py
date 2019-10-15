"""Script for benchmarking semantic segmentation results by mIoU."""
import argparse
import os

import numpy as np

from PIL import Image
from utils.metrics import iou_stats


parser = argparse.ArgumentParser(
  description='Benchmark segmentation predictions'
)
parser.add_argument('--pred_dir', type=str, default='',
                    help='/path/to/prediction.')
parser.add_argument('--gt_dir', type=str, default='',
                    help='/path/to/ground-truths')
parser.add_argument('--num_classes', type=int, default=21,
                    help='number of segmentation classes')
parser.add_argument('--string_replace', type=str, default=',',
                    help='replace the first string with the second one')
args = parser.parse_args()


assert(os.path.isdir(args.pred_dir))
assert(os.path.isdir(args.gt_dir))
tp_fn = np.zeros(args.num_classes, dtype=np.float64)
tp_fp = np.zeros(args.num_classes, dtype=np.float64)
tp = np.zeros(args.num_classes, dtype=np.float64)
for dirpath, dirnames, filenames in os.walk(args.pred_dir):
  for filename in filenames:
    predname = os.path.join(dirpath, filename)
    gtname = predname.replace(args.pred_dir, args.gt_dir)
    if args.string_replace != '':
      stra, strb = args.string_replace.split(',')
      gtname = gtname.replace(stra, strb)

    pred = np.asarray(
        Image.open(predname).convert(mode='L'),
        dtype=np.uint8)
    gt = np.asarray(
        Image.open(gtname).convert(mode='L'),
        dtype=np.uint8)
    _tp_fn, _tp_fp, _tp = iou_stats(
        pred,
        gt,
        num_classes=args.num_classes,
        background=0)

    tp_fn += _tp_fn
    tp_fp += _tp_fp
    tp += _tp

iou = tp / (tp_fn + tp_fp - tp + 1e-12) * 100.0

class_names = ['Background', 'Aero', 'Bike', 'Bird', 'Boat',
               'Bottle', 'Bus', 'Car', 'Cat', 'Chair','Cow',
               'Table', 'Dog', 'Horse' ,'MBike', 'Person',
               'Plant', 'Sheep', 'Sofa', 'Train', 'TV']

for i in range(args.num_classes):
  if i >= len(class_names):
    break
  print('class {:10s}: {:02d}, acc: {:4.4f}%'.format(
      class_names[i], i, iou[i]))
mean_iou = iou.sum() / args.num_classes
print('mean IOU: {:4.4f}%'.format(mean_iou))

mean_pixel_acc = tp.sum() / (tp_fp.sum() + 1e-12)
print('mean Pixel Acc: {:4.4f}%'.format(mean_pixel_acc))
