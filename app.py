from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("filtered_recipes_cleaned.csv")

#remove bullet points from ingredients column
data['ingredients'] = data['ingredients'].str.replace('▢', '', regex=False)
data_clean = data.dropna()

# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data_clean['clean_ingredients'])

# Combine Features (Use the dense array directly)
X_combined = X_ingredients.toarray()

# Train KNN Model
knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn.fit(X_combined)

def recommend_recipes(input_features):
    # Check for exact matches in the dataset
    exact_matches = data_clean[
        data_clean['clean_ingredients'].str.contains(input_features[0], case=False, na=False)
    ]
    if not exact_matches.empty:
        return exact_matches[['title', 'clean_ingredients', 'image']]
    
    # Transform input ingredients for KNN
    input_ingredients_transformed = vectorizer.transform([input_features[0]])
    input_combined = input_ingredients_transformed.toarray()

    # Get recommendations
    distances, indices = knn.kneighbors(input_combined)
    recommendations = data_clean.iloc[indices[0]]
    
    # Set a relevance threshold
    relevance_threshold = 1.5
    relevant_indices = np.where(distances[0] < relevance_threshold)[0]
    
    if len(relevant_indices) == 0:
        return None
    
    relevant_recommendations = recommendations.iloc[relevant_indices]
    return relevant_recommendations[['title', 'clean_ingredients', 'image']]




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
        if len(ing.strip()) < 3:  # Minimal input length to ensure validity
            return render_template(
                'index.html',
                recommendations=[],
                message="Please enter a valid ingredient list."
            )
        
        input_features = [ing]
        recommendations = recommend_recipes(input_features)
        
        if recommendations is None:
            return render_template(
                'index.html',
                recommendations=[],
                message="No relevant recipes found for your search."
            )
        
        return render_template(
            'index.html',
            recommendations=recommendations.to_dict(orient='records'),
            truncate=truncate
        )
    return render_template('index.html', recommendations=[])




@app.route('/recipe/<title>')
def view_recipe(title):
    # Find the specific recipe by name
    recipe = data_clean[data_clean['title'] == title].to_dict(orient='records')
    
    if recipe:
        # Get the first matching recipe (in case of duplicates)
        recipe = recipe[0]
        return render_template('recipe_detail.html', recipe=recipe)
    else:
        # Handle case where recipe is not found
        return "Recipe not found", 404

if __name__ == "__main__":
    app.run(debug=True)