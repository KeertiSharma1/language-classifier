## Project Overview  
This project is a machine learning-based **Language Classifier** that predicts the language of a given text. It utilizes **TF-IDF vectorization** and a **machine learning model** trained on a multilingual dataset.  

## Project Structure  
language-classifier/  
│── notebook/              # Jupyter Notebook for model training and evaluation  
│   ├── Language_Detection.ipynb  
│── data/                  # Dataset for training the model  
│   ├── Language Detection.csv  
│── models/                # Saved machine learning models  
│   ├── language_classifier.pkl  
│   ├── tfidf_vectorizer.pkl  
│── README.md              # Project documentation  

## Installation and Setup  

### Prerequisites  
Ensure you have **Python 3.x** installed along with the required libraries:  
```bash
pip install numpy pandas scikit-learn joblib jupyter

Clone the Repository
git clone https://github.com/KeertiSharma1/language-classifier.git  
cd language-classifier 
```
## Run the Jupyter Notebook
``` bash 
jupyter notebook
```
Open language_classifier.ipynb and execute all cells to train and evaluate the model.

## How It Works
- **Text Preprocessing:** Converts text into numerical format using TF-IDF Vectorization.

- **Model Training:** Trains a Naïve Bayes classifier (or another ML model) on the dataset.

- **Prediction:** Once trained, the model predicts the language of new text samples.


## Usage
### Predicting Language for New Text
Once the model is trained and saved, you can use it in Python as follows:
```bash
import joblib  

# Load the model and vectorizer  
model = joblib.load("models/language_classifier.pkl")  
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")  

# Sample input text  
new_texts = ["Bonjour, comment ça va ?", "Hola, ¿cómo estás?", "Das ist ein Beispiel."]  
new_texts_transformed = vectorizer.transform(new_texts).toarray()  

# Predict language  
predictions = model.predict(new_texts_transformed)  
print("Predicted Languages:", predictions)  
```

## Results
The trained model achieves an accuracy of 96%, demonstrating strong performance in identifying different languages.

## Future Improvements
- Enhance dataset quality for improved accuracy.

- Experiment with deep learning models like LSTMs or Transformers.

- Develop a web or mobile app for real-time language detection.

## Contributing
Contributions are welcome! You can:

- Improve the model performance.

- Add support for more languages.

- Optimize the dataset.

## License
This project is licensed under the MIT License.