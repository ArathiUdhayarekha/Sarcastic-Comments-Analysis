from data_preparation import prepare_data
from train_model import load_model, train_model, evaluate_model

def main():
    # Step 1: Prepare the data
    X_train, X_test, y_train, y_test = prepare_data('sarcastic_comments.csv')

    # Step 2: Load BERT model and tokenizer
    model, tokenizer = load_model()

    # Step 3: Train the model
    trained_model = train_model(X_train, y_train, model, tokenizer)

    # Step 4: Evaluate the model
    evaluate_model(trained_model, X_test, y_test, tokenizer)

if __name__ == "__main__":
    main()
