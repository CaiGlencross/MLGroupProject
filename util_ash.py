from string import punctuation

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# scikit-learn libraries
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle


def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1_score', 'auroc', 'precision',
                           'sensitivity', 'specificity'        
    
    Returns
    --------------------
        score  -- float, performance score
    """
    y_label = np.sign(y_pred)
    y_label[y_label == 0] = 1

    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_label)
    elif metric == "f1_score":
        return metrics.f1_score(y_true, y_label)
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred)
    elif metric == "precision":
        return metrics.precision_score(y_true, y_label)
    elif metric == "sensitivity":
        return metrics.recall_score(y_true, y_label)
    elif metric == "specificity":
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_label).ravel()
        return float(tn)/float((tn+fp))

def cv_performance(clf, X, y, kf, metric="accuracy"):
    scores = []
    for train, test in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        # use SVC.decision_function to make ``continuous-valued'' predictions
        y_pred = clf.decision_function(X_test)
        score = performance(y_test, y_pred, metric)
        if not np.isnan(score):
            scores.append(score)

    return np.array(scores).mean()


def select_param_rbf(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metric  -- string, option used to select performance measure
    
    Returns
    --------------------
        C        -- float, optimal parameter value for an RBF-kernel SVM
        gamma    -- float, optimal parameter value for an RBF-kernel SVM
    """

    print 'RBF SVM Hyperparameter Selection based on ' + str(metric) + ':'

    C_vals = 10.0 ** np.arange(-2, 2)
    G_vals = 10.0 ** np.arange(-2, 2)

    score_grid = np.zeros((len(C_vals), len(G_vals)))

    for i, c in enumerate(C_vals):
        for j, g in enumerate(G_vals):
            clf = SVC(kernel='rbf', gamma=g, C=c)
            score_grid[i, j] = cv_performance(clf, X, y, kf, metric=metric)
            print 'done with g: ', g
        print 'done with c: ', c

    i, j = np.unravel_index(score_grid.argmax(), score_grid.shape)

    return C_vals[i], G_vals[j]

def select_param_logReg(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameters of a Logistic Regression,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metric  -- string, option used to select performance measure
    
    Returns
    --------------------
        C        -- float, optimal parameter value for an RBF-kernel SVM
        gamma    -- float, optimal parameter value for an RBF-kernel SVM
    """

    print 'Logistic Regression Hyperparameter Selection based on ' + str(metric) + ':'

    C_vals = [0.25,0.5,1,1.25, 1.5,2,2.25,2.5,3,4,5]

    score_array = np.zeros(len(C_vals))

    for i, c in enumerate(C_vals):
        clf = LogisticRegression(C=c)
        score = cv_performance(clf, X, y, kf, metric=metric)
        score_array[i] = score
        print 'done with c: ', c
        print 'respective metric score: ', score

    i = score_array.argmax()

    return C_vals[i]

def determine_logreg_hyperparameters(X_training, y_training, plot=False, show=False, weight = 1):
    C_vals = [0.1,0.2,0.25,0.5,1,1.25, 1.5,2,2.25,2.5]
    c_avg_accs = []
    c_avg_f1 = []
    c_avg_auroc = []
    c_avg_prec = []
    c_avg_recall = []
    for c in C_vals:
        model = LogisticRegression(C=c, class_weight = {-1:1, 1:weight})

        #k fold validation training
        avg_accs = []
        avg_f1s = []
        avg_aurocs = []
        avg_precs = []
        avg_recalls = []
        avg_percentage = 0
        kf = KFold(n_splits = 10)
        for train_index, val_index in kf.split(X_training):
            X_train, X_val = X_training[train_index], X_training[val_index]
            y_train, y_val = y_training[train_index], y_training[val_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            avg_accs.append(model.score(X_val, y_val))
            avg_f1s.append(metrics.f1_score(y_val, y_pred))
            avg_aurocs.append(metrics.roc_auc_score(y_val, y_pred))
            avg_precs.append(metrics.precision_score(y_val, y_pred))
            avg_recalls.append(metrics.recall_score(y_val, y_pred))

        # print "average accuracy for %f k-fold validation was %f" % (c, np.mean(avg_accs))
        c_avg_accs.append(np.mean(avg_accs))
        c_avg_f1.append(np.mean(avg_f1s))
        c_avg_auroc.append(np.mean(avg_aurocs))
        c_avg_prec.append(np.mean(avg_precs))
        c_avg_recall.append(np.mean(avg_recalls))
        #if show: print "average distribution is %f" % (avg_percentage/float(10))
    if plot:
        plt.figure()
        plt.xlabel("C values")
        plt.ylabel("metric score")
        plt.plot(C_vals, c_avg_accs, label="Accuracy")
        plt.plot(C_vals, c_avg_f1, label="F1_Score")
        plt.plot(C_vals, c_avg_auroc, label="AUROC")
        plt.plot(C_vals, c_avg_prec, label="Precision")
        plt.plot(C_vals, c_avg_recall, label="Recall")
        plt.legend()
        plt.show()
    if show:
        print "optimal C for accuracy %f"  % (C_vals[np.argmax(c_avg_accs)])
        print "optimal C for f1 score %f"  % (C_vals[np.argmax(c_avg_f1)])
        print "optimal C for Auroc %f"     % (C_vals[np.argmax(c_avg_auroc)])
        print "optimal C for precision %f" % (C_vals[np.argmax(c_avg_prec)])
        print "optimal C for recall %f" % (C_vals[np.argmax(c_avg_recall)])

    # need to pick the best f1_score, within some threshold of precision

    precision_threshold = .6 #given by prof Wu
    best_index = -1
    while best_index == -1:
        max_f1 = 0
        for i in range(len(c_avg_prec)):
            if c_avg_prec[i] >= precision_threshold and c_avg_f1[i] > max_f1:
                max_f1 = c_avg_f1[i]
                best_index = i


        precision_threshold = precision_threshold-.1


    return C_vals[best_index]

def determine_DT_hyperparameters(X_training, y_training, plot=False, show=False):
    print 'starting to test for max_depth parameter...'
    depth_vals = [0.5,0.75, 1,1.5, 2,2.5, 3,3.5,4,5,6,7,8,9,10]
    #depth_vals = [10,20,30,40,50,60,70]
    d_avg_accs = []
    d_avg_f1 = []
    d_avg_auroc = []
    d_avg_prec = []
    d_avg_recall = []
    for d in depth_vals:
        print 'starting depth: ', d
        model = DTC(criterion='entropy', random_state=123, max_depth=d)

        #k fold validation training
        avg_accs = []
        avg_f1s = []
        avg_aurocs = []
        avg_precs = []
        avg_recalls = []
        avg_percentage = 0
        kf = KFold(n_splits = 4)
        for train_index, val_index in kf.split(X_training):
            X_train, X_val = X_training[train_index], X_training[val_index]
            y_train, y_val = y_training[train_index], y_training[val_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            avg_accs.append(model.score(X_val, y_val))
            avg_f1s.append(metrics.f1_score(y_val, y_pred))
            avg_aurocs.append(metrics.roc_auc_score(y_val, y_pred))
            avg_precs.append(metrics.precision_score(y_val, y_pred))
            avg_recalls.append(metrics.recall_score(y_val, y_pred))

        # print "average accuracy for %f k-fold validation was %f" % (c, np.mean(avg_accs))
        d_avg_accs.append(np.mean(avg_accs))
        d_avg_f1.append(np.mean(avg_f1s))
        d_avg_auroc.append(np.mean(avg_aurocs))
        d_avg_prec.append(np.mean(avg_precs))
        d_avg_recall.append(np.mean(avg_recalls))
        #if show: print "average distribution is %f" % (avg_percentage/float(10))
    if plot:
        plt.figure()
        plt.xlabel("max_depth values")
        plt.ylabel("metric score")
        plt.plot(depth_vals, d_avg_accs, label="Accuracy")
        plt.plot(depth_vals, d_avg_f1, label="F1_Score")
        plt.plot(depth_vals, d_avg_auroc, label="AUROC")
        plt.plot(depth_vals, d_avg_prec, label="Precision")
        plt.plot(depth_vals, d_avg_recall, label="Recall")
        plt.legend()
        plt.show()
    if show:
        print "optimal depth for accuracy %f"  % (depth_vals[np.argmax(d_avg_accs)])
        print "optimal depth for f1 score %f"  % (depth_vals[np.argmax(d_avg_f1)])
        print "optimal depth for AUROC %f"     % (depth_vals[np.argmax(d_avg_auroc)])
        print "optimal depth for precision %f" % (depth_vals[np.argmax(d_avg_prec)])
        print "optimal depth for recall %f" % (depth_vals[np.argmax(d_avg_recall)])

    # need to pick the best f1_score, within some threshold of precision

    precision_threshold = .6 #given by prof Wu
    best_index = -1
    while best_index == -1:
        max_f1 = 0
        for i in range(len(d_avg_prec)):
            if d_avg_prec[i] >= precision_threshold and d_avg_f1[i] > max_f1:
                max_f1 = d_avg_f1[i]
                best_index = i


        precision_threshold = precision_threshold-.1


    return depth_vals[best_index]




def performance_CI(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC or DummyClassifier)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
        lower        -- float, lower limit of confidence interval
        upper        -- float, upper limit of confidence interval
    """
    try:
        y_pred = clf.decision_function(X)
    except:
        y_pred = clf.predict(X)
    score = performance(y, y_pred, metric)

    n, d = X.shape
    t = 1000
    estimates = np.zeros(t)
    for t in range(t):
        X_sample = np.zeros((n, d))
        y_sample = np.zeros(n)
        for i in range(n):
            index = np.random.randint(0, n)
            X_sample[i] = X[index]
            y_sample[i] = y[index]
        try:
            y_sample_pred = clf.decision_function(X_sample)
        except:
            y_sample_pred = clf.predict(X_sample)
        sample_score = performance(y_sample, y_sample_pred, metric)
        estimates[t] = sample_score

    sorted_estimates = np.sort(estimates)
    lower = sorted_estimates[24]
    upper = sorted_estimates[974]

    return score, lower, upper


def printScores(y_true, y_pred, model, train=False):
    if train == True:
        print "train accuracy for %s was %f"      %   (model, performance(y_true, y_pred, 'accuracy'))
        print "train f1 score for %s was %f"      %   (model, performance(y_true, y_pred,'f1_score'))
        print "train precision for %s was %f"     %   (model, performance(y_true, y_pred,'precision'))
        print "train recall for %s was %f"        %   (model, performance(y_true, y_pred,'sensitivity'))
        print "train AUROC for %s was %f"         %   (model, performance(y_true, y_pred,'auroc'))

    else:
        print "test accuracy for %s was %f"      %   (model, performance(y_true, y_pred, 'accuracy'))
        print "test f1 score for %s was %f"      %   (model, performance(y_true, y_pred,'f1_score'))
        print "test precision for %s was %f"     %   (model, performance(y_true, y_pred,'precision'))
        print "test recall for %s was %f"        %   (model, performance(y_true, y_pred,'sensitivity'))
        print "test AUROC for %s was %f"         %   (model, performance(y_true, y_pred,'auroc'))




################################################################################
#  Multiclass classification
################################################################################


######################################################################
# output code functions
######################################################################

def generate_output_codes(num_classes, code_type) :
    """
    Generate output codes for multiclass classification.
    
    For one-versus-all
        num_classifiers = num_classes
        Each binary task sets one class to +1 and the rest to -1.
        R is ordered so that the positive class is along the diagonal.
    
    For one-versus-one
        num_classifiers = num_classes choose 2
        Each binary task sets one class to +1, another class to -1, and the rest to 0.
        R is ordered so that
          the first class is positive and each following class is successively negative
          the second class is positive and each following class is successively negatie
          etc
    
    Parameters
    --------------------
        num_classes     -- int, number of classes
        code_type       -- string, type of output code
                           allowable: 'ova', 'ovo'
    
    Returns
    --------------------
        R               -- numpy array of shape (num_classes, num_classifiers),
                           output code
    """
    
    ### ========== TODO : START ========== ###
    # part a: generate output codes
    # hint : initialize with np.ones(...) and np.zeros(...)

    #code for ova
    if (code_type == 'ova'):
        R = -1*np.ones((num_classes, num_classes))
        for i in range(num_classes):
            R[i,i] = 1

    #code for ovo
    elif (code_type == "ovo"):
        num_classifiers = (num_classes * (num_classes-1))/2
        R = np.zeros((num_classes, num_classifiers))

        #these tuples will be our indices (i,j) where i will be +1 and j will be -1
        tuples = []
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                tuples.append((i,j))

        # i represents the row when we should have +1 and j represents the row where we should have -1
        for n, (i,j) in enumerate(tuples):
            R[i,n] = 1
            R[j,n] = -1

    ### ========== TODO : END ========== ###
    
    return R


def load_code(filename) :
    """
    Load code from file.
    
    Parameters
    --------------------
        filename -- string, filename
    """
    
    # determine filename
    import util
    dir = os.path.dirname(util.__file__)
    f = os.path.join(dir, '..', 'data', filename)
    
    # load data
    with open(f, 'r') as fid :
        data = np.loadtxt(fid, delimiter=",")
    
    return data


def test_output_codes():
    R_act = generate_output_codes(3, 'ova')
    R_exp = np.array([[  1, -1, -1],
                      [ -1,  1, -1],
                      [ -1, -1,  1]])    
    assert (R_exp == R_act).all(), "'ova' incorrect"
    
    R_act = generate_output_codes(3, 'ovo')
    R_exp = np.array([[  1,  1,  0],
                      [ -1,  0,  1],
                      [  0, -1, -1]])
    assert (R_exp == R_act).all(), "'ovo' incorrect"


######################################################################
# loss functions
######################################################################

def compute_losses(loss_type, R, discrim_func, alpha=2) :
    """
    Given output code and distances (for one example), compute losses (for each class).
    
    hamming  : Loss  = (1 - sign(z)) / 2
    sigmoid  : Loss = 1 / (1 + exp(alpha * z))
    logistic : Loss = log(1 + exp(-alpha * z))
    
    Parameters
    --------------------
        loss_type    -- string, loss function
                        allowable: 'hamming', 'sigmoid', 'logistic'
        R            -- numpy array of shape (num_classes, num_classifiers)
                        output code
        discrim_func -- numpy array of shape (num_classifiers,)
                        distance of sample to hyperplanes, one per classifier
        alpha        -- float, parameter for sigmoid and logistic functions
    
    Returns
    --------------------
        losses       -- numpy array of shape (num_classes,), losses
    """
    
    # element-wise multiplication of matrices of shape (num_classes, num_classifiers)
    # tiled matrix created from (vertically) repeating discrim_func num_classes times
    z = R * np.tile(discrim_func, (R.shape[0],1))    # element-wise
    
    # compute losses in matrix form
    if loss_type == 'hamming' :
        losses = np.abs(1 - np.sign(z)) * 0.5
    
    elif loss_type == 'sigmoid' :
        losses = 1./(1 + np.exp(alpha * z))
    
    elif loss_type == 'logistic' :
        # compute in this way to avoid numerical issues
        # log(1 + exp(-alpha * z)) = -log(1 / (1 + exp(-alpha * z)))
        eps = np.spacing(1) # numpy spacing(1) = matlab eps
        val = 1./(1 + np.exp(-alpha * z))
        losses = -np.log(val + eps)
    
    else :
        raise Exception("Error! Unknown loss function!")
    
    # sum over losses of binary classifiers to determine loss for each class
    losses = np.sum(losses, 1) # sum over each row
    
    return losses


def hamming_losses(R, discrim_func) :
    """
    Wrapper around compute_losses for hamming loss function.
    """
    return compute_losses('hamming', R, discrim_func)


def sigmoid_losses(R, discrim_func, alpha=2) :
    """
    Wrapper around compute_losses for sigmoid loss function.
    """
    return compute_losses('sigmoid', R, discrim_func, alpha)


def logistic_losses(R, discrim_func, alpha=2) :
    """
    Wrapper around compute_losses for logistic loss function.
    """
    return compute_losses('logistic', R, discrim_func, alpha)


######################################################################
# classes
######################################################################

class MulticlassSVM :
    
    def __init__(self, R, C=1.0, kernel='linear', **kwargs) :
        """
        Multiclass SVM.
        
        Attributes
        --------------------
            R       -- numpy array of shape (num_classes, num_classifiers)
                       output code
            svms    -- list of length num_classifiers
                       binary classifiers, one for each column of R
            classes -- numpy array of shape (num_classes,) classes
        
        Parameters
        --------------------
            R       -- numpy array of shape (num_classes, num_classifiers)
                       output code
            C       -- numpy array of shape (num_classifiers,1) or float
                       penalty parameter C of the error term
            kernel  -- string, kernel type
                       see SVC documentation
            kwargs  -- additional named arguments to SVC
        """
        
        num_classes, num_classifiers = R.shape
        
        # store output code
        self.R = R
        
        # use first value of C if dimension mismatch
        try :
            if len(C) != num_classifiers :
                raise Warning("dimension mismatch between R and C " +
                                "==> using first value in C")
                C = np.ones((num_classifiers,)) * C[0]
        except :
            C = np.ones((num_classifiers,)) * C
        
        # set up and store classifier corresponding to jth column of R
        self.svms = [None for _ in xrange(num_classifiers)]
        for j in xrange(num_classifiers) :
            svm = SVC(kernel=kernel, C=C[j], **kwargs)
            self.svms[j] = svm
    
    
    def fit(self, X, y) :
        """
        Learn the multiclass classifier (based on SVMs).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), features
            y    -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        classes = np.unique(y)
        num_classes, num_classifiers = self.R.shape
        if len(classes) != num_classes :
            raise Exception('num_classes mismatched between R and data')
        self.classes = classes    # keep track for prediction
        
        ### ========== TODO : START ========== ###
        # part c: train binary classifiers
        
        # HERE IS ONE WAY (THERE MAY BE OTHER APPROACHES)
        #
        # keep two lists, pos_ndx and neg_ndx, that store indices
        #   of examples to classify as pos / neg for current binary task
        #
        # for each class C
        # a) find indices for which examples have class equal to C
        #    [use np.nonzero(CONDITION)[0]]
        # b) update pos_ndx and neg_ndx based on output code R[i,j]
        #    where i = class index, j = classifier index
        #
        # set X_train using X with pos_ndx and neg_ndx
        # set y_train using y with pos_ndx and neg_ndx
        #     y_train should contain only {+1,-1}
        #
        # train the binary classifier

        # loop through each classifier
        for j, clf in enumerate(self.svms):
            pos_ndx = []; neg_ndx = []
            #loop through each class
            for c, class_name in enumerate(classes):
                # see which examples have this class
                indices = np.nonzero(y == class_name)[0]
                indices2 = indices.tolist()

                # set pos and neg indices accordingly
                if self.R[c,j] == 1:
                    pos_ndx += indices2
                elif self.R[c,j] == -1:
                    neg_ndx += indices2

            X_train = []; y_train = []

            # update training data 
            for p in pos_ndx:
                X_train.append(X[p])
                y_train.append(1)
            for n in neg_ndx:
                X_train.append(X[n])
                y_train.append(-1)

            # train binary classifier 
            clf.fit(X_train, y_train)

        return self
        ### ========== TODO : END ========== ###
    
    
    def predict(self, X, loss_func=hamming_losses) :
        """
        Predict the optimal class.
        
        Parameters
        --------------------
            X         -- numpy array of shape (n,d), features
            loss_func -- loss function
                         allowable: hamming_losses, logistic_losses, sigmoid_losses
        
        Returns
        --------------------
            y         -- numpy array of shape (n,), predictions
        """
        
        n,d = X.shape
        num_classes, num_classifiers = self.R.shape
        
        # setup predictions
        y = np.zeros(n)
        
        ### ========== TODO : START ========== ###
        # part d: predict multiclass class
        # 
        # HERE IS ONE WAY (THERE MAY BE OTHER APPROACHES)
        #
        # for each example
        #   predict distances to hyperplanes using SVC.decision_function(...)
        #   find class with minimum loss (be sure to look up in self.classes)
        # 
        # if you have a choice between multiple occurrences of the minimum values,
        # use the index corresponding to the first occurrence

        for i in range(n):
            #compute distances to hyperplanes
            dists = np.empty(num_classifiers)
            for j in range(len(self.svms)):
                X_i_reshape = X[i,:].reshape(1,-1)
                dist = self.svms[j].decision_function(X_i_reshape)
                dists[j] = dist

            # find class with min loss
            losses = compute_losses(loss_func, self.R, dists)
            index = np.argmin(losses)
            y[i] = self.classes[index]


        ### ========== TODO : END ========== ###
        
        return y


def print_tree(decision_tree, feature_names=None, class_names=None, root=0, depth=1):
    """
    Print decision tree.
    
    Only works with decision_tree.n_outputs = 1.
    https://healthyalgorithms.com/2015/02/19/ml-in-python-getting-the-decision-tree-out-of-sklearn/
        
    Parameters
    --------------------
        decision_tree -- tree (sklearn.tree._tree.Tree or Tree)
        feature_names -- list, feature names
        class_names   -- list, class names
    """
    
    t = decision_tree
    if t.n_outputs != 1:
        raise NotImplementedError()
    
    if depth == 1:
        print 'def predict(x):'
    
    indent = '    ' * depth
    
    # determine node numbers of children
    left_child = t.children_left[root]
    right_child = t.children_right[root]
    
    # determine predicted class for this node
    values = t.value[root][0]
    class_ndx = np.argmax(values)
    if class_names is not None:
        class_str = class_names[class_ndx]
    else:
        class_str = str(class_ndx)
        
    # determine node string     
    node_str = "(node %d: impurity = %.2f, samples = %d, value = %s, class = %s)" % \
        (root, t.impurity[root], t.n_node_samples[root], values, class_str)
    
    # main code
    if left_child == tree._tree.TREE_LEAF:
        print indent + 'return %s # %s' % (class_str, node_str)
    else:
        # determine feature name
        if feature_names is not None:
            name = feature_names[t.feature[root]]
        else:
            name = "x_%d" % t.feature[root]
        
        print indent + 'if %s <= %.2f: # %s' % (name, t.threshold[root], node_str)
        print_tree(t, feature_names, class_names, root=left_child, depth=depth+1)
        
        print indent + 'else:'
        print_tree(t, feature_names, class_names, root=right_child, depth=depth+1)



# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))



