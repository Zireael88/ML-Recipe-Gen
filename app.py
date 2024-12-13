from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("filipino_recipes.csv")

#remove bullet points from ingredients column
data['ingredients'] = data['ingredients'].str.replace('▢', '', regex=False)
data_clean = data.dropna()

# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data_clean['ingredients'])

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
    recommendations = data_clean.iloc[indices[0]]
    return recommendations[['title', 'ingredients', 'image']].head(10)

# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ing = request.form['ing']
        input_features = [ing]
        recommendations = recommend_recipes(input_features)
        return render_template(
            'index.html',
            recommendations=recommendations.to_dict(orient='records'),
            truncate=truncate
        )
    return render_template('index.html', recommendations=[])

@app.route('/recipe/<recipe_name>')
def view_recipe(recipe_name):
    # Find the specific recipe by name
    recipe = data_clean[data_clean['recipe_name'] == recipe_name].to_dict(orient='records')
    
    if recipe:
        # Get the first matching recipe (in case of duplicates)
        recipe = recipe[0]
        return render_template('recipe_detail.html', recipe=recipe)
    else:
        # Handle case where recipe is not found
        return "Recipe not found", 404

if __name__ == "__main__":
    app.run(debug=True)