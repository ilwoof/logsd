# LogSD
LogSD: Detecting Anomalies from System Logs through Self-supervised Learning and Frequency-based Masking

## Description

`LogSD` is a novel semi-supervised log anomaly detection approach, which leverages a dual-network framework with three self-supervised learning tasks and frequency-based masking to better learn patterns from normal log data for anomaly detection. 
We conducted an empirical study to evaluate the effectiveness of `LogSD` on three open-source log datasets (i.e., HDFS, BGL and Spirit). The results demonstrate the effectiveness of `LogSD`. 

## Project Structure

```
├─data  # Instances for log data.
└─src├─deeploglizer├─common      # data preprocessing, data loader and common utils, etc.
     |             └─models      # Model, network modules, and loss design 
     ├─logsd_demo.py             # LogSD main entrance.     
     └─experiment_records        # expeirmental results,  model checkpoint and logs.        

```

## Environment

**Key Packages:**

PyTorch v1.11.0 + (cu11.3)

python v3.8.6

scikit-learn


## Data

The log datasets used in the paper can be found in the repo [loghub](https://github.com/logpai/loghub).
In this repository, the BGL dataset under 100logs setting is proposed for a quick hands-up.

For generating the data files, please refer to the implementation repo of [deep-loglizer](https://github.com/logpai/deep-loglizer).


## Usage

The simplest way of running LogSD is to run `python logsd.py`.


## Ablation Experiments (supplementary data)

- Ablation for Network, Masking Schemes, and Reconstruction Paradigms

|    Dataset       | HDFS MCC         | HDFS F1    | HDFS Precision | HDFS Recall | HDFS PRC   | HDFS ROC   | BGL MCC        | BGL<br>F1    | BGL Precision | BGL Recall | BGL PRC   | BGL ROC   | Spirit MCC        | Spirit F1      | Spirit Precision | Spirit Recall | Spirit PRC   | Spirit ROC   |
|------------------|------------|-------|-----------------|--------|-------|-------|------------|-------|-----------------|--------|-------|-------|------------|---------|-----------------|--------|-------|-------|
| **LogSD<sub>sng</sub>** | 0.7880   | 0.7858     | 0.8249 | 0.7692          | 0.7345  | 0.8793   | 0.9126  | 0.9152 | 0.9483         | 0.8915  | 0.9474  | 0.9935  | 0.7801     | 0.7949    | 0.6758            | 0.9650  | 0.7074     | 0.9781     |
| **LogSD<sub>srl</sub>** | 0.3872   | 0.3895     | 0.7896 | 0.2585          | 0.3071  | 0.5999   | 0.8357  | 0.8226 | 0.8833         | 0.7726  | 0.8336  | 0.9209  | 0.3335     | 0.3340    | 0.3821            | 0.3024  | 0.2590     | 0.6771     |
| **LogSD<sub>sfl</sub>** | 0.7122   | 0.7143     | 0.7731 | 0.6754          | 0.6583  | 0.8468   | 0.9095  | 0.8939 | 0.9074         | 0.8837  | 0.9377  | 0.9921  | 0.8769     | 0.8846    | 0.8761            | 0.8932  | **0.7731**     | 0.9634     |
| **LogSD<sub>srf</sub>** | 0.9101   | 0.9067     | 0.8603 | **0.9623**      | 0.9019  | 0.9977   | 0.9264  | 0.9339 | 0.9306         | 0.9380  | 0.9616  | 0.9965  | 0.8032     | 0.8012    | 0.6977            | 0.9438  | 0.7055     | 0.9824     |
| **LogSD<sub>sff</sub>** | 0.9213   | 0.9153     | 0.9333 | 0.8991          | 0.9403  | 0.9954   | **0.9596**  | 0.9534 | 0.9244         | **0.9871**  | 0.9644  | 0.9977  | 0.8930     | 0.8921    | 0.8116            | **0.9902**  | 0.7509     | 0.9837     |
| **LogSD<sub>dng</sub>** | 0.9471   | 0.9462     | 0.9491 | 0.9433          | 0.9821  | 0.9995   | 0.9384  | 0.9366 | 0.9335         | 0.9406  | 0.9489  | 0.9905  | 0.8913     | 0.8936    | 0.8367            | 0.9625  | 0.7343     | 0.9925     |
| **LogSD<sub>drl</sub>** | 0.4223   | 0.4120     | 0.8063 | 0.2769          | 0.3450  | 0.5979   | 0.8382  | 0.8371 | 0.8904         | 0.7933  | 0.8631  | 0.9515  | 0.3577     | 0.3420    | 0.3431            | 0.3456  | 0.2669     | 0.7270     |
| **LogSD<sub>dfl</sub>** | 0.7688   | 0.7597     | 0.9506 | 0.6327          | 0.7473  | 0.9458   | 0.9418  | 0.9281 | 0.9392         | 0.9173  | 0.9489  | 0.9905  | 0.8812     | 0.8886    | **0.8801**            | 0.8973  | 0.7368     | 0.9833     |
| **LogSD<sub>drf</sub>** | 0.9319   | 0.9338     | 0.9385 | 0.9292          | 0.9812  | **0.9995**   | 0.9329  | 0.9498 | 0.9470         | 0.9535  | 0.9697  | 0.9974  | 0.8807     | 0.8911    | 0.8100            | 0.9902  | 0.6930     | 0.9914     |
| **LogSD<sub>dff</sub>** | **0.9559**   | **0.9583**  | **0.9587**          | 0.9580      | **0.9840**   | 0.9993   | 0.9483  | **0.9627** | **0.9600**         | 0.9664     | **0.9716**  | **0.9977**  | **0.8954**     | **0.8957**    | 0.8386            | 0.9650        |  0.7346   | **0.9927**  |

## Sensitivity Experiments (supplementary data)

- The performance under different masking rates On the different window settings of BGL dataset 
<p align="center">
<img src=".\pic\masking_sensitivity_bgl.png" height = "200" alt="" align=center />
</p>


- The performance under different masking rates On the BGL-60logs setting

| Method |Masking Rate <br> Setting |BGL-60logs <br> MCC|BGL-60logs <br> F1 |BGL-60logs <br> Precision|BGL-60logs <br> Recall |BGL-60logs <br> PRC |BGL-60logs <br> ROC |
|--------|:---------------------- |:--------------|:-------------- |:--------------|:------------- |:-------------- |:------------- |
| &nbsp; | 0.05                  |0.9027| 0.9081        |0.9186|  0.8978       | 0.9124 | 0.9947 |
| &nbsp; | 0.1                   |0.8714| 0.8804        |0.8736|  0.8889       | 0.8563 | 0.9902 |
| &nbsp; | 0.2                   |0.8442| 0.8552        |0.8639|  0.8467       | 0.8092 | 0.9880 |
| &nbsp; | 0.3                   |0.8222| 0.8345        |0.8081|  0.8644       | 0.7431 | 0.9870 |
| LogSD  | 0.4                   |0.8871| 0.8945        |0.8662|  **0.9267**   | 0.8500 | 0.9922 |
| &nbsp; | 0.5                   |0.8756| 0.8844        |0.8934|  0.8756       | 0.8271 | 0.9911 |
| &nbsp; | 0.6                   |**0.9119**| **0.9181**|**0.9274**|  0.9089   | **0.9301** | **0.9948** |
| &nbsp; | 0.7                   |0.8949| 0.9024        |0.9116|  0.8933       | 0.9078 | 0.9938 |
| &nbsp; | 0.8                   |0.9046| 0.9113        |0.9206|  0.9022       | 0.8532 | 0.9920 |
| &nbsp; | 0.9                   |0.8430| 0.8510        |0.8192|  0.8978       | 0.7955 | 0.9867 |
| &nbsp; | {0.05, 0.1, 0.15, 0.2, 0.3} |0.8712| 0.8664  |0.8753|  0.8578       | 0.8736 | 0.9911 |
