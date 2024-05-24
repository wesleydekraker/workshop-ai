from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from dataset import get_dataset, balance_dataset, train_val_test_split


def compute_metrics(p):
    """
    This function computes the accuracy of a model's predictions.
    The accuracy is calculated as the number of correct predictions divided by the total number of predictions.
    """
    accuracy = accuracy_score(p.label_ids, p.predictions.argmax(-1))

    return {
        "accuracy": accuracy,
    }


# Load the dataset
dataset = get_dataset(".")

# Balance the dataset so that we have an equal number of vulnerable and non-vulnerable samples
dataset = balance_dataset(dataset)

# Split the dataset into train, validation, and test sets
train_dataset, val_dataset, test_dataset = train_val_test_split(dataset)

model = BertForSequenceClassification.from_pretrained('google/bert_uncased_L-2_H-128_A-2',
                                                      num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=0.2,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

# Test the model
predictions = trainer.predict(test_dataset)
print(compute_metrics(predictions))
