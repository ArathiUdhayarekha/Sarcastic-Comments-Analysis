import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path='/mnt/data/datasets.csv'):
    # Load the dataset
    data = pd.read_csv(file_path)
    return data

def preprocess_text(text):
    # Lowercase the text and remove special characters
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

def prepare_data(file_path='/mnt/data/datasets.csv'):
    # Load and preprocess data
    data = load_data(file_path)
    data['text'] = data['text'].apply(preprocess_text)

    # Split the data into train and test sets using the 'text' and 'sarcasm_ref' columns
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['sarcasm_ref'], test_size=0.2, stratify=data['sarcasm_ref']
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
