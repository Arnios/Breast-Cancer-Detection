## Model For Breast Cancer Detection

Breast cancer is the second most prevalent form of cancer worldwide. According to [World Cancer Research Fund](https://www.wcrf.org/dietandcancer/cancer-trends/breast-cancer-statistics), there were more than 2 million new cases reported in 2018 alone.

Detection of breast cancer consists of a series of tests of which a often the most critical one is a fine needle aspirate (FNA) of the breast mass.

The idea of this project is to develop a computer aided system capable of analyzing the result of the FNA test with high degree of accuracy. Minimize human errors to the best extent possible and assist medical professionals.

&nbsp;

### Dataset
____

The labelled dataset used for this project is obtained from [Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data) available under the Creative Commons ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)) license. The version of the dataset in use is V2. But the original dataset can be found in the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).

The dataset comprises of numerous parameters calculated from digitized image obtained after conducting FNA test on 569 patients. Each of the record in turn is labelled as either 'Benign' or 'Malignant'.

&nbsp;

### Evaluation Metric
___

The various model that will be developed in this project will be evaluated based on the area under their individual [Receiver Operating Characteristic (ROC)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curve.