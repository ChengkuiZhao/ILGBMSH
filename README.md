# Design-miR30-shRNA
We developed the potent shRNA target prediction model with the advanced ensemble algorithm LightGBM. Comparing with previous shRNA prediction and the deep learning method, our model achieves the best prediction.
# Workflow
![image text](https://github.com/ChengkuiZhao/Design-miR30-shRNA/blob/main/image/1.Workflow.jpg)
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
#Note
1.This code contains LightGBM algorithm and deeplearning algorithms.
2.The method suit the best for the miR30-based shRNA prediction, while also could be an option for the shRNA with general scaffold.
