from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("recipe_final.csv")

# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data['ingredients_list'])

# Combine Features (Use the dense array directly)
X_combined = X_ingredients.toarray()

# Train KNN Model
knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn.fit(X_combined)

def recommend_recipes(input_features):
    # Transform input ingredients
    input_ingredients_transformed = vectorizer.transform([input_features[0]])
    input_combined = input_ingredients_transformed.toarray()  # Use 2D array directly
    
    # Get recommendations
    distances, indices = knn.kneighbors(input_combined)
    recommendations = data.iloc[indices[0]]
    return recommendations[['recipe_name', 'ingredients_list', 'image_url']].head(10)

# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ingredients = request.form['ing']
        input_features = [ingredients]
        recommendations = recommend_recipes(input_features)
        return render_template(
            'index.html',
            recommendations=recommendations.to_dict(orient='records'),
            truncate=truncate
        )
    return render_template('index.html', recommendations=[])

if __name__ == "__main__":
    app.run(debug=True)