import numpy as np
import sys

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

train_x  = sys.argv[1]  #'/home/louiefu/Desktop/ML_HW2/X_train'
train_y  = sys.argv[2]  #'/home/louiefu/Desktop/ML_HW2/Y_train'
test_x   = sys.argv[3]  #'/home/louiefu/Desktop/ML_HW2/X_test'
out_file = sys.argv[4]  #'/home/louiefu/Desktop/ML_HW2/cheat/Y_adaboost_5.csv'

# fix random seed for reproducibility
np.random.seed(7)
X = np.genfromtxt(train_x, delimiter=',')
X = np.delete(X,0,0)
data_count, feat_count = X.shape

#read in Y_exact all -> Y_exact : (total lines in Y_train-1, 106) array
Y = np.reshape(np.genfromtxt(train_y, delimiter=','),(-1,1))
Y = np.delete(Y,0,0)
y = np.reshape(Y,(-1,))

#read in X_test all -> Test_datas :(total lines in X_test-1, 106) array
X_test = np.genfromtxt(test_x, delimiter=',')
X_test = np.delete(X_test,0,0)

#process
# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME.R", n_estimators=1200)

bdt.fit(X, y)

#joblib.dump(bdt, 'dnn_model.pkl')

#scores = cross_val_score(bdt, X, Y)
#print(scores)
print("finish training")

P = bdt.predict(X_test)
result = np.reshape(P,(-1,1))

#write to result_logistic.csv
print("start writing result_dnn.csv")
id_num = []
for i in range(result.shape[0]):
	id_num.append(i+1)
id_array = np.array(id_num)
out =  np.column_stack((id_array, result))
np.savetxt( out_file , out, delimiter=',', fmt="%i", header = 'id,label',comments='') 

print("finishing writing "+out_file+" !")
'''
plot_colors = "br"
plot_step = 0.02
class_names = "AB"

plt.figure(figsize=(10, 5))

# Plot the decision boundaries
plt.subplot(121)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")

# Plot the training points
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1],
                c=c, cmap=plt.cm.Paired,
                s=20, edgecolor='k',
                label="Class %s" % n)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary')

# Plot the two-class decision scores
twoclass_output = bdt.decision_function(X)
plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(122)
for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(twoclass_output[y == i],
             bins=10,
             range=plot_range,
             facecolor=c,
             label='Class %s' % n,
             alpha=.5,
             edgecolor='k')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc='upper right')
plt.ylabel('Samples')
plt.xlabel('Score')
plt.title('Decision Scores')

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.show()
'''
