import shap
from transformers import BertTokenizer
from train_model import load_model

def explain_predictions(model, tokenizer, comments):
    # SHAP explainer function
    explainer = shap.Explainer(lambda x: model.predict(x)[0], tokenizer)
    
    # Generate SHAP values for the provided comments
    shap_values = explainer(comments)
    
    # Plot SHAP explanations
    shap.plots.text(shap_values)

def main():
    # Load trained model and tokenizer
    model, tokenizer = load_model()
    
    # Test with some example comments
    comments = ["This is exactly what I needed... not!", "Great job!"]
    
    # Explain the predictions
    explain_predictions(model, tokenizer, comments)

if __name__ == "__main__":
    main()
