# Data Science Pipeline for Product Recommendation Prediction

This project implements a machine learning pipeline to predict whether a customer recommends a product based on their review. The dataset includes numerical, categorical, and textual data, and the model uses NLP techniques alongside traditional machine learning methods.

## Getting Started

Follow the steps below to set up the project on your local machine and run the pipeline.

### Dependencies

The following Python libraries are required for this project:
pandas numpy matplotlib seaborn scikit-learn nltk textblob


You can install all required dependencies by using the `requirements.txt` file.

### Installation

1. Clone this repository to your local machine:
   
git clone https://github.com/zoez314/Data_Science_Pipeline_Project


2. Install the required dependencies:
pip install -r requirements.txt

3. Download the dataset (`reviews.csv`) and place it in the `/data` folder.

4. Open the Jupyter Notebook `starter1.ipynb` to start working on the pipeline.

## Testing

This project does not contain unit tests but ensures the model performs well through evaluation metrics such as accuracy, precision, recall, and F1-score.

### Breakdown of Evaluation Metrics

- **Accuracy**: Measures the proportion of correct predictions.
- **Precision**: Measures the proportion of positive predictions that are actually correct.
- **Recall**: Measures the proportion of actual positive instances that were correctly identified.
- **F1-Score**: Harmonic mean of precision and recall, useful when there is an uneven class distribution.

The project includes detailed outputs in the notebook for each evaluation metric after training the model.

## Project Instructions

### Data Preprocessing

1. **The data** is first cleaned and preprocessed by separating numerical, categorical, and text features.
2. **Numerical features** like `Age` and `Positive Feedback Count` are standardized.
3. **Categorical features** like `Clothing ID`, `Division Name`, `Department Name`, and `Class Name` are one-hot encoded.
4. **Textual features** are transformed using TF-IDF vectorization for the `Review Text` and `Title` columns.

### Model Pipeline

1. The pipeline integrates preprocessing steps and model training into a seamless workflow.
2. A **Logistic Regression** model is used for binary classification (Recommended or Not Recommended).
3. The pipeline is fine-tuned using **RandomizedSearchCV** to optimize hyperparameters such as the regularization strength (`C`) and solver choice.

### Model Evaluation

The model is evaluated using cross-validation on the training data and tested using a hold-out test set. The final evaluation results are shown, including key metrics such as accuracy, precision, recall, and F1-score.

### Advanced Features

1. **Sentiment Analysis**: TextBlob is used to extract sentiment from the review text, which can be included as an additional feature for prediction.
2. **Model Fine-Tuning**: Hyperparameters are optimized using randomized search with cross-validation for better model performance.

## Built With

* [pandas](https://pandas.pydata.org/) - Data manipulation and analysis
* [scikit-learn](https://scikit-learn.org/) - Machine learning algorithms and tools
* [TextBlob](https://textblob.readthedocs.io/) - Sentiment analysis
* [matplotlib](https://matplotlib.org/) - Data visualization
* [seaborn](https://seaborn.pydata.org/) - Statistical data visualization

These libraries were used to build the project pipeline, train the model, and evaluate performance.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.
