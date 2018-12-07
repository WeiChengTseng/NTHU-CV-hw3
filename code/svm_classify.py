from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import sklearn
import pdb



def svm_classify(train_image_feats, train_labels, test_image_feats, kernel=None):
    #################################################################################
    # TODO :                                                                        #
    # This function will train a set of linear SVMs for multi-class classification  #
    # and then use the learned linear classifiers to predict the category of        #
    # every test image.                                                             # 
    #################################################################################
    ##################################################################################
    # NOTE: Some useful functions                                                    #
    # LinearSVC :                                                                    #
    #   The multi-class svm classifier                                               #
    #        e.g. LinearSVC(C= ? , class_weight=None, dual=True, fit_intercept=True, #
    #                intercept_scaling=1, loss='squared_hinge', max_iter= ?,         #
    #                multi_class= ?, penalty='l2', random_state=0, tol= ?,           #
    #                verbose=0)                                                      #
    #                                                                                #
    #             C is the penalty term of svm classifier, your performance is highly#
    #          sensitive to the value of C.                                          #
    #   Train the classifier                                                         #
    #        e.g. classifier.fit(? , ?)                                              #
    #   Predict the results                                                          #
    #        e.g. classifier.predict( ? )                                            #
    ##################################################################################
    '''
    Input : 
        train_image_feats : training images features
        train_labels : training images labels
        test_image_feats : testing images features
    Output :
        Predict labels : a list of predict labels of testing images (Dtype = String).
    '''
    if kernel is None:
        classifier = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,\
                               intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                               multi_class='ovr', penalty='l2', random_state=0, 
                               tol=1e-05, verbose=0)
    elif kernel == 'rbf':
        classifier =  sklearn.svm.SVC(kernel='rbf', 
                                      random_state=1, 
                                      gamma=1, C=10)
    classifier.fit(train_image_feats, train_labels)
    pred_label = []
    pred_label = classifier.predict(test_image_feats)

    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return pred_label
