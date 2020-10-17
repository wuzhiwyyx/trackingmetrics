# TrackingMetrics

Thanks [cheind/py-motmetrics](https://github.com/cheind/py-motmetrics) for their great job. This repository provides benchmark multiple object trackers (MOT) in Python on specified data format.

## Installation
1. Install requirements:  
`pip install -r requirements.txt`
2. Clone this repo:  
`git clone https://github.com/noble-born/TrackingMetrics.git`

## Input data
Assume you have two `.csv` files, which contains *ground_truth* and *predictions* of tracking objects on *sequence of frames*.

Each `.csv` file expected to have at least `['frame_id', 'track_id', 'x_camera', 'y_camera', 'box_w', 'box_h']` columns. Each row corresponds one object of the frame.

## Calculate metrics
Example of running:  
`python evaluateTracking.py PATH_TO_INI`  

Here is an example of `.ini` file: [sequence.ini](https://github.com/noble-born/TrackingMetrics/blob/master/sequence.ini).
Note that this benchmark will consider as many frames as  specified in `seqLength` starting from 1.

## TODO
* Add the ability to account for confidence and visibility of detections.
