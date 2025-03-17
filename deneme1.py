from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import stats
import time
import pandas as pd
import numpy as np
import os

categories = ["alt.atheism", "sci.med", "talk.politics.mideast", "rec.motorcycles"]
twenty_news = load_files(container_path='C:\\20_newsgroups', categories=categories, description=None, load_content=True,
                         shuffle=True, encoding='latin-1', decode_error='strict', random_state=42,
                         allowed_extensions=None)


def fit_in_batches(classifier, x_train, y_train, batch_size):
    if hasattr(classifier, 'partial_fit'):
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            classifier.partial_fit(x_batch, y_batch, classes=np.unique(y_train))
    else:
        classifier.fit(x_train, y_train)
    return classifier


def pre_steps(data, vectorizer, compressor, classifier):
    fold_matrix = []
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    data_vector = vectorizer.fit_transform(data.data).astype('float32').toarray()

    if compressor is not None:
        if isinstance(compressor, LDA):
            max_components = min(data_vector.shape[1], len(np.unique(data.target)) - 1)
            compressor.n_components = min(compressor.n_components, max_components)
            data_vector = compressor.fit_transform(data_vector, data.target)
            data_vector = compressor.fit_transform(data_vector, data.target)
        else:
            data_vector = compressor.fit_transform(data_vector)

    for i, (train_index, test_index) in enumerate(kf.split(data_vector, data.target)):
        data_train = data_vector[train_index]
        data_test = data_vector[test_index]
        data_train_target = data.target[train_index]
        data_test_target = data.target[test_index]
        data_train_target = np.array(data_train_target)
        data_test_target = np.array(data_test_target)

        fitted_classifier = fit_in_batches(classifier, data_train, data_train_target, batch_size=500)
        fold_matrix.append([data_train, data_test, data_train_target, data_test_target, fitted_classifier])

    return fold_matrix


# Vectorizer and compressor combinations
vectorizer = [CountVectorizer(), TfidfVectorizer()]
compressor = [None, IncrementalPCA(n_components=50, batch_size=1000), LDA(n_components=2)]
classifier = [SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None),
              LogisticRegression(max_iter=1000),
              SVC(kernel='linear', random_state=42)]
combinations = [(vec, comp, classify) for vec in vectorizer for comp in compressor for classify in classifier]

# Re-initialize classifier for each combination
all_results = []
execution_times = []
for vec, comp, classify in combinations:
    start_time = time.time()
    if isinstance(classify, SGDClassifier):
        classifier_instance = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5,
                                            tol=None)
    elif isinstance(classify, LogisticRegression):
        classifier_instance = LogisticRegression(max_iter=1000)
    elif isinstance(classify, SVC):
        classifier_instance = SVC(kernel='linear', random_state=42)

    result = pre_steps(twenty_news, vec, comp, classifier_instance)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)
    print(
        f"Evaluating Model with {vec.__class__.__name__}, {comp.__class__.__name__ if comp else 'None'}, {classify.__class__.__name__}: {execution_time: .6f} seconds")
    all_results.append(result)

# EVALUATE
all_scores = []
results = []

for i, result in enumerate(all_results):
    vec, comp, classify = combinations[i]
    print(
        f"Evaluating Model {i + 1} with {vec.__class__.__name__}, {comp.__class__.__name__ if comp else 'None'}, {classify.__class__.__name__}:")
    model_scores = []
    for j, fold in enumerate(result):
        data_train, data_test, data_train_target, data_test_target, fitted_classifier = fold
        fold_scores = cross_val_score(fitted_classifier, data_train, data_train_target, cv=5, scoring='accuracy')
        model_scores.extend(fold_scores)
        all_scores.extend(model_scores)

    mean_score = np.mean(model_scores)
    confidence_interval = stats.t.interval(0.95, len(model_scores) - 1, loc=mean_score, scale=stats.sem(model_scores))
    results.append({
        "Model": f"{vec.__class__.__name__}, {comp.__class__.__name__ if comp else 'None'}, {classify.__class__.__name__}",
        "Execution Time (s)": execution_times[i],
        "Cross-Validation Score": np.average(model_scores),
        "Mean Score": mean_score,
        "Confidence Interval": confidence_interval

    })
    print(results)
#    print(f"Average of the Cross-Validation Scores: {', '.join(f'{np.average(model_scores):.4f}')}")

mean_score = np.mean(all_scores)
confidence_interval = stats.t.interval(0.95, len(all_scores) - 1, loc=mean_score, scale=stats.sem(all_scores))
print(f"Mean score: {mean_score}\n%95 Confidence Interval: {confidence_interval}")

# DATA FRAME
df_results = pd.DataFrame(results)
data = [["Mean Score", "Confidence Interval"], [mean_score, confidence_interval]]

# EXCEL
df_results.to_excel('model_evaluation.xlsx', index=False)

print("'model_evaluation.xlsx' is created.")

# ADD NEW DATA
file_name = 'model_evaluation.xlsx'

if os.path.exists(file_name):
    existing_df = pd.read_excel(file_name)
    df_results = pd.concat([existing_df, df_results], ignore_index=True)

df_results.to_excel(file_name, index=False)

print("'model_evaluation.xlsx' is updated.")
