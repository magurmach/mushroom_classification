## Problem Description

Assess performance of decision tree and naive bayes binary classifier of
[Mashroom classifier dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification?datasetId=478).

## Procedure

`sklearn` libraries Naive Bayes and Decision Tree doesn't work with categorical
value out of the box. Pandas was used and label encoder was used to encode the
feature values as numerical code.

70/30 split was used to train and test the model. Due to lack of dediated test
dataset, better generalization error couldn't be calculated.

Besides local implementation (captured here), the data set was fed to GCP AutoMl.
To be able to feed it to AutoML, we needed to change the column names  of the
dataset in Kaggle. Guidelines for supported schema can be found
[here](https://cloud.google.com/bigquery/docs/schemas#column_names). By default
a 80/10/10 split is choses as training/validation/test dataset. 

## Performance evaluation

#### Decision tree performance:

With no limit on Decision Tree depth:

```
--------------
Decision tree performance:
Accuracy:  1
F1 score:  1
```

With `max_depth = 5`, decision tree performance:

```
--------------
Decision tree performance:
Accuracy:  0.9799015586546349
F1 score:  0.9799133472971425
```

#### Naive Bayes performance:

```
--------------
Naive Bayes performance:
Accuracy:  0.9540607054963085
F1 score:  0.953842651059095
```

#### AutoML from GCP:

Once the properly formatted dataset was uploaded, training the model in AutoML
is straightforward in GCP. No model selection or alike information was needed
as input.

|---------- |------|
| PR AUC    | 1    |
| ROC AUC   | 1    |
| Log loss  | 0    |
| F1 Score  | 1    |
| Precision | 100% |
| Recall    | 100% |

###### Correctness and other data from GCP AutoML

## Verditct

With minimal processing, both Naive Bayes and Decision Tree perfromed
relatively well.

However, decision tree seemed like a natural fit. With no depth limitation, the
decision tree yielded 100% accurance and F1 score.