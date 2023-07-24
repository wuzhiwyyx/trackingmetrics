import numpy as np
from sort import *
from tqdm import tqdm
import json
import cv2
import argparse


def visualize(data, canvas=None):
    canvas = np.zeros((160, 200, 3), dtype=np.uint8) if canvas is None else canvas
    for dd in data:
        x1, y1, x2, y2, conf = dd
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        canvas = cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
        canvas = cv2.putText(canvas, str(conf), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow('canvas', canvas)
    cv2.waitKey(0)

def save_as_csv(outs, path='gt.csv'):
    # add .csv suffix if not
    if not path.endswith('.csv'):
        path += '.csv'
    with open(path, 'w') as f:
        keys = outs.keys()
        keys = sorted(keys)
        f.write('frame_id,track_id,x_camera,y_camera,box_w,box_h\n')
        for k in tqdm(keys):
            for vv in outs[k]:
                # frame_id, trk_id, x1, y1, x2, y2
                f.write('%d,%d,%f,%f,%f,%f\n' % tuple(vv))

def load_from_coco_anno(path):
    # load data from json file with coco format
    with open(path, 'r') as f:
        data = json.load(f)
    
    outs = {}
    for anno in data['annotations']:
        xywh = anno['bbox']
        frame_id = anno['image_id']
        trk_id = anno['track_id']
        outs[frame_id] = outs.get(frame_id, [])
        # xywh to xyxy
        xyxy = [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]
        # coco frame_id starts from 1, but here we start from 0
        outs[frame_id].append([frame_id - 1, trk_id, *xyxy])
        # visualize([[*xyxy, 1.0]])
    return outs

def main(args):
    outs = load_from_coco_anno(args.anno)
    save_as_csv(outs, args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('anno', type=str, help='path of coco annotation file',
                    default='/mnt/hdd/hiber2_cocovid/hiber2_cocovid_0/annotations/instances_default.json')
    parser.add_argument('--out', type=str, default='gt.csv', help='output csv file')

    args = parser.parse_args()

    print(args)
    main(args)
