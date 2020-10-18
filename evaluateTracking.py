import argparse
import io
import logging
import os
import sys
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from configparser import ConfigParser

import motmetrics as mm


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('inifile', type=str, default='./')
    return parser.parse_args()


def prepare_dataframe(path, frames):
    df = pd.read_csv(path)
    df = df[df['frame_id'] < frames]
    df = df[['frame_id', 'track_id', 'x_camera', 'y_camera', 'box_w', 'box_h']]
    rename_dict = {'frame_id': 'FrameId', 'track_id': 'Id',
                   'x_camera': 'X', 'y_camera': 'Y',
                   'box_w': 'Width', 'box_h': 'Height'}
    df = df.rename(columns=rename_dict)
    df = df.set_index(['FrameId', 'Id'])
    df['ClassId'] = 1
    df['Visibility'] = 1
    df['Confidence'] = 1
    return df            


def compare_dataframes(gts, ts, inifile, iou=0.5):
    """Builds accumulator for each sequence."""   
    acc, ana = mm.utils.CLEAR_MOT_M(gts, ts, inifile, 'iou', distth=iou, vflag='')
    return acc, ana


def main():
    args = parse_args()
    seqIni = ConfigParser()
    seqIni.read(args.inifile, encoding='utf8')
    params = seqIni['Sequence']
    
    os.makedirs(params['log_dir'], exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', datefmt='%I:%M:%S',
            handlers=[logging.FileHandler(os.path.join(params['log_dir'], time.ctime()+'.log'), 'w', 'utf-8'),
                      logging.StreamHandler()])

    if params['solver']:
        mm.lap.default_solver = params['solver']

    logging.info('Available LAP solvers %s', str(mm.lap.available_solvers))
    logging.info('Default LAP solver \'%s\'', mm.lap.default_solver)
    logging.info('Loading files.')

    gt = prepare_dataframe(params['gt_path'], int(params['seqLength']))
    ts = prepare_dataframe(params['predictions_path'], int(params['seqLength']))

    mh = mm.metrics.create()
    st = time.time()
    acc, analysis = compare_dataframes(gt, ts, args.inifile, 1. - float(params['iou']))
    logging.info('adding frames: %.3f seconds.', time.time() - st)

    logging.info('Running metrics')
    summary = mh.compute(acc, ana=analysis, metrics=mm.metrics.motchallenge_metrics)
    logging.info(mm.io.render_summary(summary, formatters=mh.formatters,
                                      namemap=mm.io.motchallenge_metric_names))
    logging.info('Completed')


if __name__ == '__main__':
    main()
