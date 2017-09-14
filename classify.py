def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier
    clf = GaussianNB()

    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)
    
    ### use the trained classifier to predict labels for the test features
    pred = [clf.predict([features_test[i]]) for i in range(0, len(features_test))]

    correct = 0
    for ii in range(0, len(pred)):
        if pred[ii]==labels_test[ii]:
            correct+=1
            
    accuracy_manual = correct/len(labels_test)
    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = accuracy_manual
    print(clf.score(features_test, labels_test))
    from sklearn.metrics import accuracy_score
    print(accuracy_score(pred, labels_test))
    return accuracy