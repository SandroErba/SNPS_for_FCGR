import os
import numpy as np
import time
import torch
from sklearn.linear_model import LogisticRegression
from sps import handle_csv, exact_csv
from sps.digit_image import get_mnist_data
from sps.exact_csv import SNPS_exact_csv
from sps.handle_csv import SNPS_csv, extend_csv, ensemble_csv
from sps.config import Config
from sps.snp_system import SNPSystem
from sklearn.svm import LinearSVC



def launch_dataset(x,y):
    #3/4 in training, 1/4 in test, it's important that the test set does not
    #contain elements from the training
    np.random.seed(123)
    indices = np.random.choice(x.shape[0],size=int(x.shape[0]/4*3),replace=False)
    x_train = x[indices]
    y_train = y[indices]
    x_test = np.delete(x, indices, axis=0)
    y_test = np.delete(y, indices)

    print(x_train.shape,y_train.shape)
    print(x_test.shape,y_test.shape)
    t=time.time()
    #example_direct(x_train, y_train, x_test, y_test) #optional

    SNPS_csv() #create the csv for the SNPS
    svm, logreg = train_SNPS(x_train, y_train)
    print("-> Done the training procedure")

    ensemble_accuracy = test_SNPS(x_test, y_test, svm, logreg)
    handle_csv.save_results(ensemble_accuracy, time.time()-t)
    return ensemble_accuracy

def launch_mnist():
    x_train, y_train, x_test, y_test = get_mnist_data()
    t=time.time()
    example_direct(x_train, y_train, x_test, y_test) #optional

    SNPS_csv() #create the csv for the SNPS
    svm, logreg = train_SNPS(x_train, y_train)
    print("-> Done the training procedure")

    ensemble_accuracy = test_SNPS(x_test, y_test, svm, logreg)
    handle_csv.save_results(ensemble_accuracy, time.time()-t)
    return ensemble_accuracy





def train_SNPS(x_train, y_train):

    snps = SNPSystem(Config.TRAIN_SIZE, Config.TRAIN_SIZE + 5, True,"TRAIN")

    SNPS_exact_csv() #create the csv for the SNPS with rules in the exact form
    snps.load_neurons_from_csv("csv/" + Config.CSV_EXACT_NAME)

    snps.spike_train = x_train
    snps.start()
    pooling = snps.pooling_image.T

    np.save("/tmp/pooling_snp.npy", pooling)
    print("saved to /tmp/pooling_snp.npy")

    return train_external_models(pooling, y_train)



def train_external_models(charges, y_train):

    # Support Vector Machine
    svm = LinearSVC(C=Config.SVM_C, max_iter=100000)
    svm.fit(charges, y_train)
    print("SVM done")
    
    # Logistic Regression
    logreg = LogisticRegression(solver="lbfgs", max_iter=100000)
    logreg.fit(charges, y_train)
    print("LogReg done")
    
    return svm, logreg



def test_SNPS(x_test, y_test, svm, logreg):

    compare_performance(x_test, y_test, svm, logreg) #optional

    ensemble_pred = ensemble_and_test(x_test, svm.coef_, logreg.coef_, get_importance(svm.coef_), get_importance(logreg.coef_))

    print("---> y_test label corrette:", y_test, "prediction of ensemble model:", ensemble_pred)

    ensemble_accuracy = np.mean(ensemble_pred == y_test)

    print("SNPS ensemble accuracy with importance:", ensemble_accuracy)

    return ensemble_accuracy



def ensemble_and_test(x_test, svm_w, logreg_w, svm_imp, logreg_imp):

    snps = SNPSystem(Config.TEST_SIZE, Config.TEST_SIZE + 5, True,"TEST")
    svm_q = ternarize_matrix(svm_w.T)
    logreg_q = ternarize_matrix(logreg_w.T)

    snps.spike_train = x_test

    extended_path = exact_csv.ensemble_exact_csv(np.array(svm_q), np.array(logreg_q), svm_imp, logreg_imp)  #rules in exact form

    snps.load_neurons_from_csv(extended_path)
    snps.start()

    print("-> Done the test phase")
    print("---> entire prediction map:", snps.charge_map_prediction)
    y_pred = np.argmax(snps.charge_map_prediction, axis=0)

    return y_pred



def get_importance(w):
    imp = compute_neuron_importance(w)
    if Config.DISCRETIZE_METHOD == 1: multipliers = discretize_percentile(imp)
    else: multipliers = discretize_proportional(imp) #Config.DISCRETIZE == 2
    return multipliers


def compute_neuron_importance(w):
    #Magnitude: neurons with larger overall weights across classes are considered more influential
    if Config.IMPORTANCE_METHOD == 1: imp = np.linalg.norm(w, axis=0)
    # Weight range: neurons whose weights vary more between classes are considered more discriminative
    else: imp = np.max(w, axis=0) - np.min(w, axis=0) #Config.IMPORTANCE_METHOD == 2
    imp = imp / max(imp.max(), 1e-8) #Normalize in range [0:1]
    return imp

def discretize_percentile(imp): #method 1 - percentile based importance
    p25 = np.percentile(imp, 25)   # first quartile
    p75 = np.percentile(imp, 75)   # third quartile
    multipliers = np.ones_like(imp)  # default multiplier = 1 (low importance)
    multipliers[imp > p75] = 3       # top 25% most important neurons
    multipliers[(imp > p25) & (imp <= p75)] = 2  # middle 50%
    multipliers[imp <= p25] = 1      # bottom 25%
    return multipliers.astype(int) # convert to integer

def discretize_proportional(imp): #method 2 - proportional based importance
    multipliers = 1 + np.round(imp * Config.DISC_RANGE)
    return multipliers.astype(int) # convert to integer


def ternarize_matrix(w):
    # matrix quantization for last layer of SNPS: Transform from real values to {-1,0,1}
    if Config.TERNARIZE_METHOD == 1: q = ternarize_percentile(w, Config.M_SPARSITY, Config.M_POSITIVE) # Percentile-based
    else: q = ternarize_threshold(w, Config.M_THRESHOLD) # Threshold-based

    np.save("ternary_matrix.npy", q)

    return q


#for more info see https://www.emergentmind.com/topics/ternary-weight-networks-twns
def ternarize_percentile(w, p_zero, p_pos):
    """
    Ternary quantization {-1,0,1} using fixed percentiles per column.
    The smallest weights become -1, the largest become +1, the rest are 0.
    """
    p_neg = 1 - p_zero - p_pos
    w_q = np.zeros_like(w)
    n_rows, n_cols = w.shape

    for c in range(n_cols):
        col = w[:, c]
        sorted_idx = np.argsort(col)  # indices of sorted weights
        n_neg = int(p_neg * n_rows)
        n_zero = int(p_zero * n_rows)

        w_q[sorted_idx[:n_neg], c] = -1        # smallest weights
        w_q[sorted_idx[n_neg:n_neg+n_zero], c] = 0
        w_q[sorted_idx[n_neg+n_zero:], c] = 1  # largest weights

    return w_q

def ternarize_threshold(w, k):
    """
    Ternary quantization {-1,0,1} using a column-wise threshold.
    Weights larger than k * mean(|w|) become ±1, others become 0.
    """
    w_q = np.zeros_like(w)
    n_rows, n_cols = w.shape

    for c in range(n_cols):
        col = w[:, c]
        t = k * np.mean(np.abs(col))  # threshold for this column
        w_q[:, c] = np.where(col > t, 1, np.where(col < -t, -1, 0))  # ternary mapping

    return w_q



def example_direct(x_train, y_train, x_test, y_test):
    '''
    Compare the performance of the baseline models, launched directly on input images, without SNPS feature extraction
    Act as a lower bound, see dotted blue and orange lines in graphs/accuracy_graphs
    '''
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    svm = LinearSVC(C=Config.SVM_C, max_iter=100000)
    svm.fit(x_train, y_train)

    #Logistic Regression
    logreg = LogisticRegression(
        solver="lbfgs",
        max_iter=10000
    )
    logreg.fit(x_train, y_train)

    print("svm direct accuracy", svm.score(x_test, y_test))
    print("logreg direct accuracy", logreg.score(x_test, y_test))


def compare_performance(x_test, y_test, svm, logreg):
    '''
    Run and check performance of all the other sub-networks.
    See the accuracy graph in graphs/accuracy_graphs
    '''
    #-------------------------Testing the svm on SNPS-------------------------
    snps_svm_pred, _, svm_scores = extend_and_test(x_test,"svm", svm.coef_, None)
    snps_svm_accuracy = np.mean(snps_svm_pred == y_test)
    print("SNPS svm accuracy:", snps_svm_accuracy) # A SNPS only with svm and without importance -> green line

    #add importance
    snps_imp_svm_pred, features, _ = extend_and_test(x_test,"svm_imp", svm.coef_, get_importance(svm.coef_))
    snps_imp_svm_accuracy = np.mean(snps_imp_svm_pred == y_test)
    print("SNPS svm with importance accuracy:", snps_imp_svm_accuracy) # A SNPS only with svm and importance -> purple line

    features_int_pos = np.maximum(features // 4, 0) #extract feature from images

    #real weights SVM
    raw_svm_accuracy = svm.score(features_int_pos.T, y_test)
    print("real weights svm accuracy:", raw_svm_accuracy) # svm with real weights applied on the third layer of the SNPS -> lightgreen dotted line

    #-------------------------Testing the logreg on SNPS-------------------------
    snps_lr_pred, _, lr_scores = extend_and_test(x_test,"lr", logreg.coef_, None)
    snps_lr_accuracy = np.mean(snps_lr_pred == y_test)
    print("SNPS logreg accuracy:", snps_lr_accuracy) # A SNPS only with logreg and without importance -> red line

    #add importance
    snps_imp_lr_pred, _, _ = extend_and_test(x_test,"lr_imp", logreg.coef_, get_importance(logreg.coef_))
    snps_imp_lr_accuracy = np.mean(snps_imp_lr_pred == y_test)
    print("SNPS logreg with importance accuracy:", snps_imp_lr_accuracy) # A SNPS only with logreg and importance -> brown line

    #real weights LOGREG
    raw_lr_accuracy = logreg.score(features_int_pos.T, y_test)
    print("real weights logreg accuracy:", raw_lr_accuracy) # logreg with real weights applied on the third layer of the SNPS -> lightblue dotted line

    #------------------combined charge------------
    # sum charge of the models without importance
    sum_pred = svm_scores.T + lr_scores.T
    sum_labels = np.argmax(sum_pred, axis=0)
    ens_accuracy = np.mean(sum_labels == y_test)
    print("SNPS ensemble accuracy:", ens_accuracy) # The equivalent of a SNPS with an ensemble (both svm and logreg) without importance -> pink line



    return snps_svm_accuracy, snps_lr_accuracy, snps_imp_svm_accuracy, snps_imp_lr_accuracy, ens_accuracy, raw_svm_accuracy, raw_lr_accuracy


def extend_and_test(x_test, method, w, multipliers):
    snps = SNPSystem(Config.TEST_SIZE, Config.TEST_SIZE + 5, True)
    snps.spike_train = x_test
    q = ternarize_matrix(w.T)
    extended_path = extend_csv("csv/" + Config.CSV_NAME, np.array(q), method, multipliers)
    snps.load_neurons_from_csv(extended_path)
    snps.start()
    y_pred = np.argmax(snps.charge_map_prediction, axis=0)

    return y_pred, snps.pooling_image, snps.charge_map_prediction.T
