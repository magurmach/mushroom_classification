## Problem Description

Assess performance of decision tree and naive bayes binary classifier of
[Mashroom classifier dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification?datasetId=478).

## Procedure

`sklearn` libraries Naive Bayes and Decision Tree doesn't work with categorical
value out of the box. Pandas was used and label encoder was used to encode the
feature values as numerical code.

With minimal processing, both Naive Bayes and Decision Tree perfromed
relatively well.

However, decision tree seemed like a natural fit.

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

## Verditct
