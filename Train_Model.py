from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

def load_model():
    # Load the BERT model for sequence classification and the tokenizer
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

def train_model(X_train, y_train, model, tokenizer):
    # Tokenize the training data
    train_encodings = tokenizer(
        list(X_train), truncation=True, padding=True, max_length=200, return_tensors='tf'
    )

    # Convert labels to tensor
    train_labels = tf.convert_to_tensor(y_train.values)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), 
                  loss=model.compute_loss, metrics=['accuracy'])

    # Train the model
    model.fit(train_encodings, train_labels, epochs=3, batch_size=16)
    
    return model

def evaluate_model(model, X_test, y_test, tokenizer):
    # Tokenize the test data
    test_encodings = tokenizer(
        list(X_test), truncation=True, padding=True, max_length=200, return_tensors='tf'
    )

    # Get predictions from the model
    test_predictions = model.predict(test_encodings)[0]
    y_pred = tf.argmax(test_predictions, axis=1).numpy()

    # Print evaluation metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Example usage
    from data_preparation import prepare_data
    X_train, X_test, y_train, y_test = prepare_data('sarcastic_comments.csv')
    model, tokenizer = load_model()
    model = train_model(X_train, y_train, model, tokenizer)
    evaluate_model(model, X_test, y_test, tokenizer)
