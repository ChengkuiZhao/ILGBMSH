# ILGBMSH
We developed the intepretable shRNA target prediction model with the advanced ensemble algorithm LightGBM. Comparing with previous shRNA prediction and the deep learning method, our model achieves the best prediction.
# Workflow
![image text](https://github.com/ChengkuiZhao/Design-miR30-shRNA/blob/main/image/1.Workflow.png)
# Installation
git clone https://github.com/ChengkuiZhao/Design-miR30-shRNA
# Requirement
lightgbm 3.2.0

pandas 1.0.1

sklearn 0.22.1

matplotlib 3.1.3

numpy 1.20.3

scipy 1.4.1

torch 1.8.1+cu101
# Online Tool
The online tool is available at http://shrna.unicar-therapy.com
# Note
1. This code contains LightGBM algorithm and deeplearning algorithms.
2. The method suit the best for the miR30-based shRNA prediction, while also could be an option for the shRNA with general scaffold.
## Citation
If you use this data, tool or code, please considering citing:
Zhao C, Xu N, Tan J, Cheng Q, Xie W, Xu J, Wei Z, Ye J, Yu L, Feng W. ILGBMSH: an interpretable classification model for the shRNA target prediction with ensemble learning algorithm. Brief Bioinform. 2022 Nov 19;23(6):bbac429. doi: 10.1093/bib/bbac429. PMID: 36184189.
        
        
        
        
