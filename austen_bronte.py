from sklearn.datasets import load_files
from transformers import BertTokenizer, TFBertModel
from types import SimpleNamespace
from sklearn.utils import Bunch
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import f1_score
from scipy import stats, sparse
import time
import pandas as pd
import numpy as np
import os

# data = load_files(container_path="C:\\Users\\rnhhy\\Desktop\\text\\writer_data", categories=None, description=None, load_content=True,
#                  shuffle=True, encoding='latin-1', decode_error='strict', random_state=42,
#                  allowed_extensions=None)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = TFBertModel.from_pretrained('bert-base-uncased')

"""
# Load the data in paragraphs of 4 sentences with labeling the books with corresponding path index
def load_books(paths, sentence_number=2):
    data_paragraphs = []
    labels = []

    for label, path in enumerate(paths):
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip() for line in f]

        sentences = []
        current_sentence = []

        for line in lines:
            # Check if it is the end of the sentence
            if line.strip() == "":
                # If it is join the lines the sentence stored in current_sentences to form a complete sentence
                if current_sentence:
                    sentences.append(" ".join(current_sentence).strip())
                    current_sentence = []
            else:
                current_sentence.append(line.strip())
        # In case it is end of the page and it does not end with an empty line
        if current_sentence:
            sentences.append(" ".join(current_sentence).strip())
            current_sentence = []
        # Join the sentences into paragraphs, four sentences per one paragraph
        for i in range(0, len(sentences) - sentence_number + 1, sentence_number):
            paragraph = sentences[i: i + sentence_number]
            paragraph_text = " ".join(paragraph)
            # Consider short paragraphs and augment those shorter than 10 words to the next paragraph
            if len(paragraph_text.split()) <= 10 and (i + 2 * sentence_number) <= len(sentences):
                paragraph += sentences[i + sentence_number: i + 2 *sentence_number]
                paragraph_text = " ". join(paragraph)
            data_paragraphs.append(paragraph_text)
            labels.append(label)

        # Store the data in sklearn format
        data = SimpleNamespace()
        data.data = data_paragraphs
        data.target = labels

    return data

"""
# Load the data in paragraphs of 4 sentences with labeling the books with corresponding path index
def load_books(paths, sentence_number=2,  min_words=10):
    data_paragraphs = []
    labels = []

    for label, files in paths.items():
        for file_path in files:
            if not os.path.isfile(file_path):
                print(f"Warning: {file_path} does not exist.")
                continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        sentences = sent_tokenize(text)
        sentences = (s.strip() for s in sentences if s.strip())
        sentences = list(sentences)

        i = 0
        while i <= len(sentences) - sentence_number:
            paragraph = sentences[i: i + sentence_number]
            paragraph_text = " ".join(paragraph)
            # Consider short paragraphs and augment those shorter than 10 words to the next paragraph
            if len(paragraph_text.split()) <= min_words and (i + 2 * sentence_number) <= len(sentences):
                paragraph += sentences[i + 2 * sentence_number]
                paragraph_text = " ".join(paragraph)
                i += sentence_number # Avoid overlap
            
            data_paragraphs.append(paragraph_text)
            labels.append(label)
            i += sentence_number
        # Return the data in sklearn format
        return Bunch(data=data_paragraphs, target=labels)

# Classify the data in batches for memory error
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


# Adjust the size by adding zeros if the vector doesn't satisfy the required size of the classifier
def feature_size(x, target_dim):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[1] < target_dim:
        zeros = np.zeros((x.shape[0], target_dim - x.shape[1]))
        x_resize = np.hstack((x, zeros))
        return x_resize
    return x


# Return embeddings with [CLS] token
# Add [PAD] token to sentences, resize if too long and return tensors for bert inputs
# Vectorize using bert
# Pick one vector per sentence representing the whole meaning
# Return embedding in numpy format since it is lighter and faster than tensorflow, also sklearn, xgboost expects numpy
# Apply this proces in batches, and return the arrray of the combination of the all batches
def cls_embeddings(sentences, batch_size=20):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batches = sentences[i:i+batch_size]
        inputs = tokenizer(batches, padding=True,
                           truncation=True, return_tensors='tf')
        outputs = bert(inputs)
        # Contains the hidden representations for each token in each sequence of the batch
        embedding = outputs.last_hidden_state[:, 0, :]
        embeddings.append(embedding.numpy())

    return np.vstack(embeddings)


# Vectorize the data
# Split the data into train and test and compress separately
# Classify after adjusting the size of the data vector
# Store the matrices in fold_matrix list
def pre_steps(data, data_vector, compressor, classifier):

    fold_matrix = []
    
    if compressor is not None:
            if sparse.issparse(data_vector):
                data_vector = data_vector.toarray()
#            if isinstance(compressor, LDA):
#                data_vector = compressor.fit_transform(
#                    data_vector, data.target)
            else:
                data_vector = compressor.fit_transform(data_vector)
#        print(f"Compressed shape: {data_vector.shape}")

    kf = KFold(n_splits=10, random_state=42, shuffle=True)
#    print(f"Vectorized shape: {data_vector.shape}")
    if sparse.issparse(data_vector):
        pass # Keep sparce for tf-idf
    else:
        data_vector = data_vector.astype(np.float32)

    for train_index, test_index in kf.split(data_vector, data.target):

        data_train = data_vector[train_index]
        data_test = data_vector[test_index]
        data_train_target = data.target[train_index]
        data_test_target = data.target[test_index]
        
        data_train = feature_size(data_train, 5000)
        data_test = feature_size(data_test, 5000)
        
        classifier_copy = classifier.__class__(**classifier.get_params())
        try:
            fitted_classifier = fit_in_batches(
                classifier_copy, data_train, data_train_target, batch_size=500)

            fold_matrix.append(
                [data_train, data_test, data_train_target, data_test_target, fitted_classifier])
    #        if sparse.issparse(data_vector):
    #            data_vector = StandardScaler(with_mean=False).fit_transform(data_vector)
    #        else:
    #            data_vector = StandardScaler().fit_transform(data_vector)
        except Exception as e:
            print(f"Error in training: {e}")
            continue
    return fold_matrix


if __name__ == "__main__":
    # Load data
    data = load_books({
        0 : [
            "C:\\Users\\rnhhy\\Desktop\\text\\writer_data\\austen\\001.txt",
            "C:\\Users\\rnhhy\\Desktop\\text\\writer_data\\austen\\002.txt",
            "C:\\Users\\rnhhy\\Desktop\\text\\writer_data\\austen\\003.txt"
            ],
        1 : [
            "C:\\Users\\rnhhy\\Desktop\\text\\writer_data\\bronte\\001.txt",
            "C:\\Users\\rnhhy\\Desktop\\text\\writer_data\\bronte\\002.txt",
            "C:\\Users\\rnhhy\\Desktop\\text\\writer_data\\bronte\\003.txt"
            ]})

    # Avoid empty documents
    valid_indices = [i for i, doc in enumerate(data.data) if doc.strip()]
    if len(valid_indices) < len(data.data):
        data.data = [data.data[i] for i in valid_indices]
        data.taget = data.target[valid_indices]

    # Vectorizer and compressor combinations
    vectorizer = TfidfVectorizer(max_features=3000)
    tfidf_vector = vectorizer.fit_transform(
        data.data)
    bert_vector = cls_embeddings(data.data)

    combinations = [(tfidf_vector, None, XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=100)),
                    (bert_vector, None, XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=100))]

    # Re-initialize classifier for each combination
    all_results = []
    execution_times = []
    for vec, comp, classify in combinations:
        start_time = time.time()

        try:
            result = pre_steps(data, vec, comp, classify)
            print(f"pre_steps returned {len(result)} folds.")

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
    all_cross_scores = []
    all_f1_scores = []
    results = []

    for i, result in enumerate(all_results):
        vec, comp, classify = combinations[i]
        vec_name = vec.__class__.__name__ if not sparse.issparse(vec) else 'TfidfVectorizer'
                
        print(f"\nEvaluating Model {i+1}: {vec_name}, {comp.__class__.__name__ if comp else 'None'}, {classify.__class__.__name__}")
        cross_scores = []
        f1_scores = []

        for fold in result:
            # Unpack (train, test, target, model)
            data_train, data_test, data_train_target, data_test_target, fitted_classifier = fold

            predictions = fitted_classifier.predict(data_test)

            # Cross-validation score
            fold_scores = cross_val_score(
                fitted_classifier, data_train, data_train_target, cv=5, scoring='accuracy')
            cross_scores.extend(fold_scores)
            all_cross_scores.extend(cross_scores)

            # F1 score calculation
            f1 = f1_score(data_test_target, predictions, average="weighted")
            f1_scores.append(f1)
            all_f1_scores.extend(f1_scores)

        mean_cross_score = np.mean(cross_scores) if cross_scores else None
        mean_f1_score = np.mean(f1_scores) if f1_scores else None

        cross_confidence_interval = stats.t.interval(
            0.95, len(cross_scores)-1, loc=mean_cross_score, scale=stats.sem(cross_scores)) if len(cross_scores) > 1 else (None, None)
        f1_confidence_interval = stats.t.interval(
            0.95, len(f1_scores)-1, loc=mean_f1_score, scale=stats.sem(f1_scores)) if len(f1_scores) > 1 else (None, None)

        results.append({
            "Model": f"{vec.__class__.__name__}, {comp.__class__.__name__ if comp else 'None'}, {classify.__class__.__name__}",
            "Execution Time (s)": execution_times[i],
            "Cross-Validation Score": np.average(cross_scores),
            "Mean Score": mean_cross_score,
            "Cross Validation Confidence Interval": cross_confidence_interval,
            "Mean F1 Score": mean_f1_score,
            "Cross Validation Confidence Interval": f1_confidence_interval
        })
        print(results)
    #    print(f"Average of the Cross-Validation Scores: {', '.join(f'{np.average(model_scores):.4f}')}")

    mean_cross_score = np.mean(all_cross_scores)
    cross_confidence_interval = stats.t.interval(
        0.95, len(all_cross_scores)-1, loc=mean_cross_score, scale=stats.sem(all_cross_scores))
    print(
        f"Mean score: {mean_cross_score}\n%95 Confidence Interval: {cross_confidence_interval}")

    mean_f1 = np.mean(all_f1_scores)
    f1_confidence_interval = stats.t.interval(
        0.95, len(all_f1_scores)-1, loc=mean_f1, scale=stats.sem(all_f1_scores))
    print(
        f"Mean F1 Score: {mean_f1}\n95% Confidence Interval: {f1_confidence_interval}")

    # DATA FRAME
    df_results = pd.DataFrame(results)
    data = [["Mean Score", "Cross Validation Confidence Interval", "Mean F1 Score", "F1 Confidence Interval"],
            [mean_cross_score, cross_confidence_interval, mean_f1, f1_confidence_interval]]

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