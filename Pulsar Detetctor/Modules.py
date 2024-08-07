def optimised_model(X, y, Min_sample_leaf, N_estimators, Max_depth, Min_sample_split, test_size):
    ''' 
    This function will optimize the paramters of a RandomForestClassifier using SearchGridCV. 
    Params:
    X ~ Feature
    y ~ Target
    Range 1 ~ Minimum sample leaf
    Range 2 ~ N Estimators
    Range 3 ~ Max Depth
    Range 4 ~ Minimum sample split
    The output of the function will be F1, precision, recall and a confusion matrix. The metrics are all given as a ratio of the                         optimised:unoptimised models
    '''
    # Imports 
    import pandas  as pd
    from sklearn.model_selection import GridSearchCV
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    import seaborn as sns  


    #Make sure that test size is between 0.1 and 1
    if test_size < 0.1 or test_size > 1:
        raise ValueError("Test size must be between 0.1 and 1.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        
        
    param_grid = {
#        'min_samples_leaf': Min_sample_leaf,
#        'n_estimators': N_estimators,
        'criterion': ['gini', 'entropy'],
        'max_depth' : Max_depth,
        'min_samples_split': Min_sample_split
    }

    # make a GridSearchCV instance
    grid = GridSearchCV(RandomForestClassifier(), param_grid)

    # Fit the GridSearchCV object to the training data
    grid.fit(X_train, y_train)

    best = grid.best_params_
    print(best)

    #Plot with optimised params
    model1 = RandomForestClassifier(**best)

    # Train the model on the test dataset
    model1.fit(X_test, y_test)
    # # Predict on the test set
    y_pred = model1.predict(X_test)

    # Calc the metrics (3)
    accuracy2 = accuracy_score(y_test, y_pred)
    precision2 = precision_score(y_test, y_pred, pos_label=0)
    recall2 = recall_score(y_test, y_pred, pos_label=0)
    f1_2 = f1_score(y_test, y_pred, pos_label=0)

    mat = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(9, 9))
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
    plt.xlabel('true label')
    plt.ylabel('predicted label');

    # compare data from default and optimised model
    print(f' Accuracy ratio is {accuracy2}')
    print(f' Precision ratio is {precision2}')
    print(f' Recall ratio is {recall2}')
    print(f' F1 ratio is {f1_2}')
def learning_curve_plot(X, y, estimator, num_train):
    '''
    Learning curve plots training and test scores against the size of the training set. 
    Params:
    
    X ~ feature
    y ~ target 
    num_train ~ size of steps between 0.1 and 1 of data size

    Returns a learning curve plot with labels and legend.
    '''
    # Imports 
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import learning_curve
    import numpy as np
    import matplotlib.pyplot as plt

#    estimator = LogisticRegression()

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, 
                                                         cv=5, 
                                                         n_jobs=1, 
                                                         train_sizes = np.linspace(0.1 , 1, num_train)) 
    # we want the train size to vary between 10% and 100% in whatever steps we wish to test 
    
    #Calculate the train and test scores
    train_score_mean = np.mean(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)
    
    plt.grid()
    plt.title('Learning curve')
    plt.plot(train_sizes, train_score_mean, 'o-', label='Training score')
    plt.xlabel('Training size')
    plt.plot(train_sizes, test_score_mean, 'o-', color='red', label='Test score')
    plt.ylabel('Score')
    plt.ylim([0.5, 1])
    plt.legend(loc='best')
    plt.show
