import pickle
import cv2
import numpy as np
from sort import *
from tqdm import tqdm

def visualize(data, canvas=None):
    canvas = np.zeros((160, 200, 3), dtype=np.uint8) if canvas is None else canvas
    for dd in data:
        x1, y1, x2, y2, conf = dd
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        canvas = cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
        canvas = cv2.putText(canvas, str(conf), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow('canvas', canvas)
    cv2.waitKey(0)

def nms(dets, thresh):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        dets: (np.ndarray): shape (N, 5), (x1, y1, x2, y2, conf)
        thresh: (float): intersection over union threshold
    Returns:
        np.ndarray: shape (N, 5), (x1, y1, x2, y2, conf)
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2] + x1
    y2 = dets[:, 3] + y1
    scores = dets[:, -1]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # sort by confidence
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]  # pick maxmum confidence
        keep.append(i)
        # compute IoU of the first order with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0., xx2 - xx1 + 1.)
        h = np.maximum(0., yy2 - yy1 + 1.)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # keep non-overlapping boxes
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return dets[keep]

def tracker_format(dets):
    """convert dets to tracker format

    Args:
        dets (np.ndarray): shape (N, 5), (x1, y1, x2, y2, conf)
    Returns:
        np.ndarray: shape (N, 5), (cls, cx, cy, w, h)
    """
    # apply non-maximum-suppression
    if dets.shape[0] > 0:
        dets = nms(dets, 0.3)
    return dets
    # # convert to (cx, cy, w, h)
    # cx = (dets[:, 0] + dets[:, 2]) / 2
    # cy = (dets[:, 1] + dets[:, 3]) / 2
    # w = dets[:, 2] - dets[:, 0]
    # h = dets[:, 3] - dets[:, 1]
    # cls = np.zeros((dets.shape[0], 1))
    # return np.concatenate([cls, cx[:, None], cy[:, None], w[:, None], h[:, None]], axis=1)

def load_pred_from_pkl(args):
    with open(args.pred, 'rb') as f:
        data = pickle.load(f)
    return data

def save_as_csv(data, thresh, mot_tracker, path='output.csv'):
    # add .csv suffix if not
    if not path.endswith('.csv'):
        path += '.csv'
    with open(path, 'w') as f:
        f.write('frame_id,track_id,x_camera,y_camera,box_w,box_h\n')
        for frame_id, d in enumerate(tqdm(data)):
            batch = d[0]
            filtered = batch[batch[:, -1] > thresh]
            tracked, tracked_dets_id = mot_tracker.update(tracker_format(filtered), return_idx=True)
            for dd in tracked:
                x1, y1, x2, y2, trk_id = dd
                # xyxy to xywh
                x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

                f.write('%d,%d,%f,%f,%f,%f\n' % (frame_id, trk_id, x1, y1, x2, y2))

def main(args):
    data = load_pred_from_pkl(args)
    mot_tracker = Sort(max_age=10, iou_threshold=0.3)
    if args.vis:
        for d in data:
            visualize(d)
    save_as_csv(data, args.thresh, mot_tracker, args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pred', type=str, help='path of prediction pkl file',
                    default='/home/virgil/workspace/detbase/faster_rcnn_metric.pkl')
    parser.add_argument('--out', type=str, default='out.csv', help='output csv file')
    parser.add_argument('--thresh', type=float, default=0.2, help='threshold to filer out preds with low confidence')
    parser.add_argument('--vis', action='store_true', help='whether to visualize the result')

    args = parser.parse_args()

    print(args)
    main(args)