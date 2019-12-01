# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.utils.multiclass import unique_labels
#
#
# def c_classification_report(y_true, y_pred, target_names):
#     report = classification_report(y_true, y_pred, target_names=target_names)
#     np.savetxt('report.txt', [report], fmt='%s')
#
#
# def c_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
#     if not title:
#         if normalize:
#             title = 'Normalized confusion matrix'
#         else:
#             title = 'Confusion matrix, without normalization'
#
#     cm = confusion_matrix(y_true, y_pred)
#     classes = classes[unique_labels(y_true, y_pred)]
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            xticklabels=classes, yticklabels=classes,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     return ax
#
#
# #np.set_printoptions(precision=2)
# #c_confusion_matrix([0, 1, 0, 0, 1], [0, 1, 1, 0, 2], np.array(["true", "false", "else"]), normalize=True)
# #plt.tight_layout()
# #plt.savefig("confusionmatrix.png")
#
# c_classification_report([0, 1, 0, 0, 1], [0, 1, 1, 0, 2], np.array(["true", "false", "else"]))

###old ^^^    linear regression, polynomial regression, support vector regression, support verctor regression, random forest regression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import export_graphviz
from IPython.display import Image
from subprocess import call


def evalLinReg(df, colP, colT, x_train, y_train, model):
    print("mean_absolute_error: " + str(mean_absolute_error(df[colT], df[colP])))
    print("mean_squared_error: " + str(mean_squared_error(df[colT], df[colP])))
    print("r2_score: " + str(r2_score(df[colT], df[colP])))
    plt.scatter(x_train, y_train, color="red")
    plt.plot(x_train, model.predict(x_train), color="green")
    plt.title("Set Title")
    plt.xlabel("Set Label")
    plt.ylabel("Set Label")
    plt.show()


def evalPolyReg(df, colP, colT, X, model):
    print("mean_absolute_error: " + str(mean_absolute_error(df[colT], df[colP])))
    print("mean_squared_error: " + str(mean_squared_error(df[colT], df[colP])))
    print("r2_score: " + str(r2_score(df[colT], df[colP])))
    plt.scatter(X, df[colT], color='red')
    plt.plot(X, df[colP], color='blue')
    plt.title("Set Title")
    plt.xlabel("Set Label")
    plt.ylabel("Set Label")
    plt.show()


def evalSVR(df, colP, colT):
    print("mean_absolute_error: " + str(mean_absolute_error(df[colT], df[colP])))
    print("mean_squared_error: " + str(mean_squared_error(df[colT], df[colP])))
    print("r2_score: " + str(r2_score(df[colT], df[colP])))
    print("to be visualized")


def evalRFR(df, colP, colT, model, features, days):
    print("mean_absolute_error: " + str(mean_absolute_error(df[colT], df[colP])))
    print("mean_squared_error: " + str(mean_squared_error(df[colT], df[colP])))
    print("r2_score: " + str(r2_score(df[colT], df[colP])))
    for estimator in model.estimators:
        export_graphviz(estimator, out_file='tree.dot',
                    feature_names=features,
                    class_names=days,
                    rounded=True, proportion=False,
                    precision=2, filled=True)
        call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
        # Display in jupyter notebook
        Image(filename='tree.png')


