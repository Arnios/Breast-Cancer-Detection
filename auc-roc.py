##################################### Performance Evaluation with AUC-ROC Method ######################################

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

random_probabilities = [0 for i in range(len(Y_Test))]
fpr_noskill, tpr_noskill, _ = roc_curve(Y_Test, random_probabilities, pos_label = 1) # ROC curve for no skill approach

fpr_classifier_1, tpr_classifier_1, thresholds_classifier_1 = roc_curve(Y_Test, classifier_1.predict(X_Test), pos_label = 1) # ROC curve for the Artifical Neural Network
auc_score_classifier_1 = roc_auc_score(Y_Test, classifier_1.predict(X_Test)) # AUC Score for Artifical Neural Network
print('AUC Score of model : %.4f' %auc_score_classifier_1) # AUC Score = 98.29%

# Plot

plt.title('AUC-ROC Curve')

plt.plot(fpr_noskill, tpr_noskill, linestyle = '--', color = 'blue', label = 'No Skill')
plt.plot(fpr_classifier_1, tpr_classifier_1, linestyle = '--', color = 'red', label = 'Model')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.show()