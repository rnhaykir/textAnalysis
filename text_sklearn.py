from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import stats, sparse
import time
import pandas as pd
import numpy as np
import os

data = load_files(container_path="C:\\Users\\rnhhy\\Desktop\\20_newsgroups" , categories=None, description=None, load_content=True,
                  shuffle=True, encoding='latin-1', decode_error='strict', random_state=42,
                  allowed_extensions=None)


def fit_in_batches(classifier, x_train, y_train, batch_size):
    u_classes = np.unique(y_train)
    if hasattr(classifier, 'partial_fit'):
        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            if i == 0:
                classifier.partial_fit(x_batch, y_batch, classes=u_classes)
            else:
                classifier.partial_fit(x_batch, y_batch)
    else:
        classifier.fit(x_train, y_train)
    return classifier

def pre_steps(data, vectorizer, compressor, classifier):

    fold_matrix = []
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    data_vector = vectorizer.fit_transform(
        data.data)
    print(f"Vectorized shape: {data_vector.shape}")
    data_vector = data_vector.astype(np.float32)

    for train_index, test_index in kf.split(data_vector, data.target):
        if compressor is not None:
            if sparse.issparse(data_vector):
                data_vector = data_vector.toarray()
            if isinstance(compressor, LDA):
                data_vector = compressor.fit_transform(data_vector, data.target)
            else:
                data_vector = compressor.fit_transform(data_vector)
        print(f"Compressed shape: {data_vector.shape}")
        data_vector = feature_size(data_vector, 5000)
        
        data_train = data_vector[train_index]
        data_test = data_vector[test_index]  
        data_train_target = data.target[train_index]
        data_test_target = data.target[test_index]

        fitted_classifier = fit_in_batches(
            classifier, data_train, data_train_target, batch_size=500)

        fold_matrix.append([data_train, data_test, data_train_target, data_test_target, fitted_classifier])
#        if sparse.issparse(data_vector):
#            data_vector = StandardScaler(with_mean=False).fit_transform(data_vector)
#        else:
#            data_vector = StandardScaler().fit_transform(data_vector)
    return fold_matrix

def feature_size(x, target_dim):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[1] < target_dim:
        zeros = np.zeros((x.shape[0], target_dim - x.shape[1]))
        x_resize = np.hstack((x, zeros))
        return x_resize
    return x

# Vectorizer and compressor combinations
vectorizer = [CountVectorizer(max_features=3000), TfidfVectorizer(max_features=3000)]
compressor = [None, IncrementalPCA(
    n_components=500, batch_size=1000), LDA(n_components=2)]
classifier = [SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5000, tol=None), LogisticRegression(solver="saga", max_iter=10000, class_weight="balanced"),
              SVC(kernel='linear', random_state=42)]
combinations = [(vec, comp, classify)
                for vec in vectorizer for comp in compressor for classify in classifier]

# Re-initialize classifier for each combination
all_results = []
execution_times = []
for vec, comp, classify in combinations:
    start_time = time.time()
    
    try:
        result = pre_steps(data, vec, comp, classify)
    except MemoryError:
        print(f"MemoryError occurred with {vec}, {comp}, {classify}")
        continue
    except Exception as e:
        print(f"Error: {e} with {vec}, {comp}, {classify}")
        continue

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
    for fold in result:
        data_train, data_test, data_train_target, data_test_target, fitted_classifier = fold
        fold_scores = cross_val_score(
            fitted_classifier, data_train, data_train_target, cv=5, scoring='accuracy')
        model_scores.extend(fold_scores)
        all_scores.extend(model_scores)

    mean_score = np.mean(model_scores)
    confidence_interval = stats.t.interval(
        0.95, len(model_scores)-1, loc=mean_score, scale=stats.sem(model_scores))
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
confidence_interval = stats.t.interval(
    0.95, len(all_scores)-1, loc=mean_score, scale=stats.sem(all_scores))
print(
    f"Mean score: {mean_score}\n%95 Confidence Interval: {confidence_interval}")

# DATA FRAME
df_results = pd.DataFrame(results)
data = [["Mean Score", "Confidence Interval"],
        [mean_score, confidence_interval]]

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