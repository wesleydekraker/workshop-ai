# Deep Learning for Vulnerability Detection

This project aims to apply deep learning techniques for vulnerability detection in PHP code. We use a BERT model to classify PHP code as either vulnerable or safe. The model is trained on a synthetic dataset of PHP code samples.

## Dataset

The dataset used in this project is the "2015-10-27-php-vulnerability-test-suite" from the NIST Software Assurance Reference Dataset (SARD). The dataset can be downloaded from the following link: https://samate.nist.gov/SARD/downloads/test-suites/2015-10-27-php-vulnerability-test-suite.zip

After downloading the dataset, unzip it and place the resulting directory in the same directory as the Python scripts for this project. The main scripts will automatically load the dataset and preprocess it for use in the BERT model.

## Requirements

This project requires Python 3.10 and several Python packages, including PyTorch, Transformers, and scikit-learn. The required packages can be installed using pip:

```
pip install -r requirements.txt
```

## Running the Project

To run the project, simply execute the `main.py` script:

```
python main.py
```

This will train the BERT model on the dataset, evaluate its performance, and print the accuracy of the model.

## Docker

A Dockerfile is also provided for running the project in a Docker container. To build the Docker image, use the following command:

```
docker build -t deep-learning-vulnerability-detection .
```

Then, to run the Docker container, use the following command:

```
docker run deep-learning-vulnerability-detection
```

This will start the training process. At the end of the process, the model's accuracy will be printed.

## License

This project is open source and available under the MIT License.