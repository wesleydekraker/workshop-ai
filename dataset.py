import os
import re

import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('google/bert_uncased_L-2_H-128_A-2')


def _tokenize(php_code):
    """
    Tokenization in this context is the process of converting the PHP code into a format that the BERT model can
    understand. The BERT tokenizer breaks down the PHP code into smaller pieces, or "tokens", and then converts these
    tokens into numerical representations. These numerical representations can then be used as input for the BERT model.
    """
    return tokenizer(php_code, return_tensors='pt', padding='max_length', truncation=True, max_length=50)


def _remove_php_comments(php_code):
    """
    In our synthetic dataset, comments are used to indicate whether the code is vulnerable or not. However, such
    explicit indicators are not present in real-world code. If we train our model with these comments, it might learn
    to rely on them to classify the code, which would not be effective when it encounters real-world code
    that doesn't contain these comments.
    Therefore, we remove all comments from our synthetic dataset. This includes single line comments starting with //\
    or #, multiline comments enclosed in /* and */, and HTML comments enclosed in <!-- and -->.
    """

    patterns = [r'//.*?$', r'#.*?$', r'/\*.*?\*/', r'<!--.*?-->']
    flags = [re.MULTILINE, re.MULTILINE, re.DOTALL, re.DOTALL]

    for pattern, flag in zip(patterns, flags):
        php_code = re.sub(pattern, '', php_code, flags=flag)

    return php_code


def _remove_synthetic_words(php_code):
    """
    In our synthetic dataset, certain words such as 'safe' and 'unsafe' may indicate whether the code is vulnerable or
    not. However, such explicit indicators are not typically present in real-world code. If we train our model with
    these words, it might learn to rely on them to classify the code, which would not be effective when it encounters
    real-world code that doesn't contain these words. Therefore, we remove these words from our synthetic dataset.
    """

    return re.sub(r'tainted|sanitized|unsafe|safe|UserData', 'unknown', php_code)


def _remove_synthetic_indicators(php_code):
    """
    This function is used to remove synthetic indicators from the PHP code. It first removes all comments from the code.
    Then it removes certain words that may indicate whether the code is vulnerable or not.
    The purpose of removing these indicators is to make the model more effective when it encounters real-world code
    that doesn't contain these indicators.
    """

    php_code = _remove_php_comments(php_code)
    return _remove_synthetic_words(php_code)


def _process_sample(php_code):
    """
    This function processes a given PHP code sample for use in a BERT model.
    The function first determines a label for the PHP code, setting it to 0 if the code is deemed safe (contains the
    string "Safe sample") and 1 if it is vulnerable. Then, it removes synthetic indicators from the PHP code.
    After removing these indicators, the function tokenizes the PHP code using a pre-trained BERT tokenizer. The
    function returns a dictionary containing the tokenized input IDs, the attention mask, and the label.
    """

    label = 0 if "Safe sample" in php_code else 1
    php_code = _remove_synthetic_indicators(php_code)
    encoding = _tokenize(php_code)

    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'labels': torch.tensor(label)
    }


def get_dataset(dataset_path):
    """
    This function retrieves a dataset of vulnerable/non-vulnerable samples for a machine learning task. It walks through
    a given directory path, checks for PHP files with SQL vulnerabilities, reads the file content and processes it for
    use in a BERT model.
    """

    print("Loading dataset...")
    dataset = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            # Check if the file ends with .php and is SQL vulnerability
            if file.endswith('.php') and "CWE_89_":
                # Construct the full file path
                file_path = os.path.join(root, file)

                with open(file_path, "r") as f:
                    php_code = f.read()
                    dataset.append(_process_sample(php_code))

    if len(dataset) == 0:
        raise Exception(f"No PHP samples found in '{dataset_path}'")

    return dataset


def balance_dataset(dataset):
    """
    This function balances a binary labeled dataset by reducing the number of samples in the class with more samples
    to match the number of samples in the class with fewer samples.
    Balancing a dataset is important because machine learning models can be biased towards the class with more samples,
    resulting in poorer performance on the class with fewer samples.
    """

    # Balance the dataset by removing samples
    safe_samples = [sample for sample in dataset if sample["labels"] == 0]
    unsafe_samples = [sample for sample in dataset if sample["labels"] == 1]

    # Balance the classes
    min_samples = min(len(safe_samples), len(unsafe_samples))
    safe_samples = safe_samples[:min_samples]
    unsafe_samples = unsafe_samples[:min_samples]

    return safe_samples + unsafe_samples


def train_val_test_split(dataset):
    """
    This function splits a given dataset into three subsets: training, validation, and testing datasets.
    This function is important because it helps to prevent overfitting and underfitting when training machine learning
    models. The training dataset is used to train the model, the validation dataset is used to tune the model and
    select the best parameters, and the testing dataset is used to evaluate the final performance of the model.
    """

    train_test_dataset, val_dataset = train_test_split(dataset, test_size=0.1)
    train_dataset, test_dataset = train_test_split(train_test_dataset, test_size=1/9)

    return train_dataset, val_dataset, test_dataset

