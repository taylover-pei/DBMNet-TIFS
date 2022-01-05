# DBMNet
The pytorch implementation of [**Dual-Branch Meta-learning Network with Distribution Alignment for Face Anti-spoofing**](https://ieeexplore.ieee.org/document/9646915/keywords#keywords).

The motivation of the proposed DBMNet method:
<div align=center>
<img src="https://github.com/taylover-pei/DBMNet-TIFS/blob/main/article/Motivation.png" width="700" height="200" />
</div>

The network architecture of the proposed DBMNet method:
<div align=center>
<img src="https://github.com/taylover-pei/DBMNet-TIFS/blob/main/article/Architecture.png" width="750" height="345" />
</div>

An overview of the proposed DBMNet method:

<div align=center>
<img src="https://github.com/taylover-pei/DBMNet-TIFS/blob/main/article/Overview.png" width="500" height="300" />
</div>

## Congifuration Environment
- python 3.6 
- pytorch 0.4 
- torchvision 0.2
- cuda 8.0

## Pre-training

### **Dataset.** 

Download the OULU-NPU, CASIA-FASD, Idiap Replay-Attack, MSU-MFSD, 3DMAD, and HKBU-MARs datasets.

### **Face Detection and Face Alignment.** 

[MTCNN algorithm](https://ieeexplore.ieee.org/abstract/document/7553523) is utilized for face detection and face alignment. All the detected faces are normalized to 256$\times$256$\times$3, where only RGB channels are utilized for training. The exact codes that we used can be found [here](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection).

Put the processed frames in the path `$root/mtcnn_processed_data/`

To be specific, we utilize the MTCNN algorithm to process every frame of each video and then utilize the `sample_frames` function in the `utils/utils.py` to sample frames during training.

### **Label Generation.** 

Take the `DBMNet_ICM_to_O` experiment for example. Move to the folder `$root/cross_database_testing/DBMNet_ICM_to_O/data_label` and generate the data label list:
```python
python generate_label.py
```

## Training

Move to the folder `$root/cross_database_testing/DBMNet_ICM_to_O` and just run like this:
```python
python train_DBMNet.py
```

The file `config.py` contains the hype-parameters used during training.

## Testing

Move to the folder `$root/cross_database_testing/DBMNet_ICM_to_O` and run like this:
```python
python dg_test.py
```

## Supplementary
In this section, we provide a more detailed supplement to the experiment of our original paper.

### **Experimental Setting**
Since most of the methods focusing on the cross-database face anti-spoofing do not point out how exactly the threshold is tuned and how the HTER is calculated, we provide two sets of comparisons, i.e., **the idealized comparisons** (the results are not shown in our paper) and **the more realistic comparisons** (the results shown in our paper). 

For the idealized setting, we obtain the threshold by computing the EER directly on the target testing set, and then, compute the HTER based on the threshold. For the realistic setting, we first compute the EER on the source validation set to get the threshold. And then, we utilize the threshold to compute FAR and FRR based on the target testing set. Finally, HTER is calculated by the means of FAR and FRR. 

|Testing Task|Training Set|Validation Set|Testing Set|
|------|---|---|---|
|O&C&I to M of idealized comparisons|The original training set and testing set of OULU, CASIA, and Replay databases|None|The original training set and testing set of MSU database|
|O&C&I to M of more realistic comparisons|The original training set and testing set of OULU and Replay databases, the original training set of CASIA database|The original validation set of OULU and Replay databases, the original testing set of CASIA database|The original training set and testing set of MSU database|

To be specific, in **the idealized comparisons**, we reproduce the state-of-the-art RFMetaFAS and SSDG method, **tuning the threshold directly on the target testing set**. As shown the first line of the above table, we list the training set, validation set, and testing set of the O&C&I to M testing task in the idealized comparisons. Moreover, in **the more realistic comparisons**, we re-run the state-of-the-art methods, i.e., RFMetaFAS and SSDG, **tuning the threshold on the source validation set**. As shown the second line of the above table, we list the training set, validation set, and testing set of the O&C&I to M testing task in the more realistic comparisons.

In the subsections blow, we provide full comparison results relating to the idealized setting as well as the more realistic 
setting.

### **Cross Database Testing.** 

The comparison results of the idealized setting:
<table>
   <tr>
      <td rowspan="2">Method</td>
      <td colspan="2">O&C&I to M</td>
      <td colspan="2">O&M&I to C</td>
      <td colspan="2">O&C&M to I</td>
      <td colspan="2">I&C&M to O</td>
   </tr>
   <tr>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
   </tr>
   <tr>
      <td>RFMetaFAS</td>
      <td>8.57</td>
      <td>95.91</td>
      <td>11.44</td>
      <td>94.47</td>
      <td>16.5</td>
      <td>91.72</td>
      <td>17.78</td>
      <td>89.89</td>
   </tr>
   <tr>
      <td>SSDG-R</td>
      <td>7.38</td>
      <td>97.17</td>
      <td>10.44</td>
      <td>95.94</td>
      <td>11.71</td>
      <td><b>96.59</b></td>
      <td>15.61</td>
      <td>91.54</td>
   </tr>
   <tr>
      <td><b>DBMNet</b></td>
      <td><b>4.52</b></td>
      <td><b>98.78</b></td>
      <td><b>8.67</b></td>
      <td><b>96.52</b></td>
      <td><b>10</b></td>
      <td>96.28</td>
      <td><b>11.42</b></td>
      <td><b>95.14</b></td>
   </tr>
</table>

The comparison results of the more realistic setting:
<table>
   <tr>
      <td rowspan="2">Method</td>
      <td colspan="2">O&C&I to M</td>
      <td colspan="2">O&M&I to C</td>
      <td colspan="2">O&C&M to I</td>
      <td colspan="2">I&C&M to O</td>
   </tr>
   <tr>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
   </tr>
   <tr>
      <td>RFMetaFAS</td>
      <td>10.24</td>
      <td>95.2</td>
      <td>18.67</td>
      <td>93.69</td>
      <td>22.64</td>
      <td>75.43</td>
      <td>19.39</td>
      <td>88.75</td>
   </tr>
   <tr>
      <td>SSDG-R</td>
      <td>11.19</td>
      <td>93.69</td>
      <td>14.78</td>
      <td><b>94.74</b></td>
      <td>16.64</td>
      <td><b>91.93</b></td>
      <td>24.29</td>
      <td>88.72</td>
   </tr>
   <tr>
      <td><b>DBMNet</b></td>
      <td><b>7.86</b></td>
      <td><b>96.54</b></td>
      <td><b>14</b></td>
      <td>94.58</td>
      <td><b>16.42</b></td>
      <td>90.88</td>
      <td><b>17.59</b></td>
      <td><b>90.92</b></td>
   </tr>
</table>

### **Cross Database Testing of Limited Source Domains.** 

The comparison results of the idealized setting:
<table>
   <tr>
      <td rowspan="2">Method</td>
      <td colspan="2">M&I to C</td>
      <td colspan="2">M&I to O</td>
   </tr>
   <tr>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
   </tr>
   <tr>
      <td>RFMetaFAS</td>
      <td>29.33</td>
      <td>74.03</td>
      <td>33.19</td>
      <td>74.63</td>
   </tr>
   <tr>
      <td>SSDG-R</td>
      <td>18.67</td>
      <td>86.67</td>
      <td>23.19</td>
      <td>84.73</td>
   </tr>
   <tr>
      <td><b>DBMNet</b></td>
      <td><b>16.78</b></td>
      <td><b>89.6</b></td>
      <td><b>20.56</b></td>
      <td><b>88.33</b></td>
   </tr>
</table>

The comparison results of the more realistic setting:
<table>
   <tr>
      <td rowspan="2">Method</td>
      <td colspan="2">M&I to C</td>
      <td colspan="2">M&I to O</td>
   </tr>
   <tr>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
   </tr>
   <tr>
      <td>RFMetaFAS</td>
      <td>22.56</td>
      <td>84.37</td>
      <td>35.73</td>
      <td>77.65</td>
   </tr>
   <tr>
      <td>SSDG-R</td>
      <td>18.89</td>
      <td><b>91.35</b></td>
      <td>26.44</td>
      <td>78.14</td>
   </tr>
   <tr>
      <td><b>DBMNet</b></td>
      <td><b>16.89</b></td>
      <td>90.65</td>
      <td><b>23.73</b></td>
      <td><b>84.33</b></td>
   </tr>
</table>

### **Cross Database Cross Type Testing.** 

The comparison results of the idealized setting:
<table>
   <tr>
      <td rowspan="2">Method</td>
      <td colspan="2">O&C&I&M to D</td>
      <td colspan="2">O&C&I&M to H</td>
   </tr>
   <tr>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
   </tr>
   <tr>
      <td>RFMetaFAS</td>
      <td>5.88</td>
      <td>98.94</td>
      <td>25.38</td>
      <td>65.69</td>
   </tr>
   <tr>
      <td>SSDG-R</td>
      <td>5.88</td>
      <td>98.54</td>
      <td>24.77</td>
      <td>66.50</td>
   </tr>
   <tr>
      <td><b>DBMNet </b></td>
      <td><b>0.88 </b></td>
      <td><b>99.97 </b></td>
      <td><b>10.46 </b></td>
      <td><b>96.74 </b></td>
   </tr>
</table>

The comparison results of the more realistic setting:
<table>
   <tr>
      <td rowspan="2">Method</td>
      <td colspan="2">O&C&I&M to D</td>
      <td colspan="2">O&C&I&M to H</td>
   </tr>
   <tr>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
   </tr>
   <tr>
      <td>RFMetaFAS</td>
      <td>5.88</td>
      <td>98.35</td>
      <td>41.67</td>
      <td>81.54</td>
   </tr>
   <tr>
      <td>SSDG-R</td>
      <td>6.77</td>
      <td>98.42</td>
      <td>32.50</td>
      <td>73.68</td>
   </tr>
   <tr>
      <td><b>DBMNet </b></td>
      <td><b>0.59 </b></td>
      <td><b>99.50 </b></td>
      <td><b>20.83 </b></td>
      <td><b>92.26 </b></td>
   </tr>
</table>

### **Intra Database Testing.** 

The comparison results of the idealized setting:
<table>
   <tr>
      <td rowspan="2">Method</td>
      <td colspan="2">O&C&I&M</td>
   </tr>
   <tr>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
   </tr>
   <tr>
      <td>RFMetaFAS</td>
      <td>2.82</td>
      <td>99.45</td>
   </tr>
   <tr>
      <td>SSDG-R</td>
      <td>0.89</td>
      <td>99.90</td>
   </tr>
   <tr>
      <td><b>DBMNet </b></td>
      <td><b>0.08 </b></td>
      <td><b>99.99 </b></td>
   </tr>
</table>

The comparison results of the more realistic setting:
<table>
   <tr>
      <td rowspan="2">Method</td>
      <td colspan="2">O&C&I&M</td>
   </tr>
   <tr>
      <td>HTER(%)</td>
      <td>AUC(%)</td>
   </tr>
   <tr>
      <td>RFMetaFAS</td>
      <td>3.00</td>
      <td>99.62</td>
   </tr>
   <tr>
      <td>SSDG-R</td>
      <td>1.57</td>
      <td>99.78</td>
   </tr>
   <tr>
      <td><b>DBMNet </b></td>
      <td><b>1.48 </b></td>
      <td><b>99.83 </b></td>
   </tr>
</table>

### **Conclusion.** 

As can be seen, all the results of the RFMetaFAS, SSDG, and our DBMNet degrade when a more realistic model training process is performed. The possible reason lies in two aspects: firstly, the threshold is tuned on the source validation set instead of the target testing set; secondly, there are fewer training samples than before shown in the first table. **As a result, there is still a long way to go in the research of cross-database face anti-spoofing for better generalization.**

To be specific, we only release the code and provide the comparison results in our paper based on the more realistic setting.

## Citation
Please cite our paper if the code is helpful to your research.
```
@ARTICLE{9646915,  
   author={Jia, Yunpei and Zhang, Jie and Shan, Shiguang},  
   journal={IEEE Transactions on Information Forensics and Security},   
   title={Dual-Branch Meta-Learning Network With Distribution Alignment for Face Anti-Spoofing},   
   year={2022},  
   volume={17},
   pages={138-151},  
   doi={10.1109/TIFS.2021.3134869}}
```




