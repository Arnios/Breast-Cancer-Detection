# Deep Learning Model For Breast Cancer Detection

Breast cancer is the most commonly occurring form of cancer among women and the second most prevalent overall in the world. According to [World Cancer Research Fund](https://www.wcrf.org/dietandcancer/cancer-trends/breast-cancer-statistics), there were more than 2 million new cases reported in 2018 alone. Detection of breast cancer consists of a series of tests of which often the most critical one is a fine needle aspirate (FNA) of the breast mass. In an FNA test, a hollow needle attached to a syringe is used to withdraw the necessary amount of tissue from the area of suspicion on the breast mass of the patient. Then features are computed from a digitized image of the extracted sample. These features describe various characteristics of the cell nuclei present in the image in a 3-dimensional space. The idea behind this project was to develop a computer aided system capable of analyzing the result of the FNA test with high degree of accuracy. Minimize human errors to the best extent possible and assist medical professionals.

## Dataset

The dataset used for this research was obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29). The dataset is composed of 30 features calculated from the digitized image obtained from the sample extracted via an FNA test on 569 patients. Each record is labelled as either 'B' or 'M' standing for Benign and Malignant respectively, to indicate the individual diagnosis. In the dataset, we found 357 records with an outcome of ‘B’ whereas 212 records have ‘M’ as the outcome. Thus, the distribution of the output data is slightly skewed in favour of 'Benign' outcomes.

## Evaluation Metric

The models developed and tested for this project were measured on their predictive accuracy on the validation set on which they were asked to run their prediction.

## Models

We first performed a comparative analysis of the prediction accuracy achieved by the various classification models on the Wisconsin Breast Cancer Diagnosis dataset. Then we tested three different Deep Learning models on the same dataset and recorded the median accuracy they obtained.

## Training

We trained our model in the same manner as before by consistently decreasing the training set concentration from 90% to 50% of the entire dataset. While increasing the concentration of the corresponding test from 10% to 50%. We maintained a step of 5% in between each iteration. In the case of the Deep Learning models, in addition to the above, we also trained and tested them with various batch sizes (BS). Starting with an initial value of 32 and then halving it till we reached a batch size of 1. Furthermore, we recorded only the median accuracy achieved by the models over 100 iterations in each of the aforementioned scenarios, as Deep Learning models are often susceptible to volatile performances and can yield varying predictive accuracy, even under the same set of circumstances.

## Disclaimer

This is still work in porgress, so some minor issues and changes are highly possible. If you encoutered any problems, I am more that happy to accept pull requests with changes. Please do not hesitate to submit your issues and/or pull requests.