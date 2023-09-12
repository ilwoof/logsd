# LogSD
LogSD: Detecting Anomalies from System Logs through Self-supervised Learning and Frequency-based Masking

## Description

`LogSD` is a novel semi-supervised log anomaly detection approach, which leverages a dual-network framework with three self-supervised learning tasks and frequency-based masking to better learn patterns from normal log data for anomaly detection. 
We conducted an empirical study to evaluate the effectiveness of `LogSD` on three open-source log datasets (i.e., HDFS, BGL and Spirit). The results demonstrate the effectiveness of `LogSD`. 

## Project Structure

```
├─data  # Instances for log data.
└─src├─deeploglizer├─  # PLELog main entrance.
     |             ├─common      # data preprocessing, data loader and common utils, etc.
     |             └─models      # Model, network modules, and loss design 
     ├─logsd_demo.py             # LogSD main entrance.     
     └─experiment_records        # expeirmental results,  model checkpoint and logs.        

```

## Environment

**Key Packages:**

PyTorch v1.11.0 + (cu11.3)

python v3.8.7

scikit-learn


## Data

The log datasets used in the paper can be found in the repo [loghub](https://github.com/logpai/loghub).
In this repository, the BGL dataset under 100logs setting is proposed for a quick hands-up.

For generating the data files, please refer to the implementation repo of [deep-loglizer](https://github.com/logpai/deep-loglizer).


## Usage

The simplest way of running LogSD is to run `python logsd.py`.

