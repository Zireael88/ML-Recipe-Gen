from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import re

app = Flask(__name__)

# Add custom regex filter
@app.template_filter('regex_replace')
def regex_replace(s, find, replace):
    """A non-optimal implementation of a regex filter"""
    return re.sub(find, replace, s)

# Load the dataset
data = pd.read_csv("filtered_recipes_cleaned.csv")

# Clean ingredients column (remove bullet points or special characters if necessary)
data['ingredients'] = data['ingredients'].str.replace('▢', '', regex=False)  # Clean ingredients
data_clean = data.dropna()  # Remove rows with missing ingredients

# Vectorize the ingredients using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_ingredients = vectorizer.fit_transform(data_clean['clean_ingredients'])

# Function to recommend recipes based on input ingredients
# Fit KNN model
knn = NearestNeighbors(n_neighbors=18, metric='cosine')
knn.fit(X_ingredients)

# Function to recommend recipes using KNN
def recommend_recipes(input_ingredients):
    input_vector = vectorizer.transform([input_ingredients])
    distances, indices = knn.kneighbors(input_vector)
    
    # Define a distance threshold (adjust as needed)
    threshold = 0.7  # Lower values mean stricter matching
    
    # Filter recommendations based on distance
    if distances[0][0] > threshold:
        return pd.DataFrame(columns=['title', 'clean_ingredients', 'image'])  # Return empty DataFrame if no close match
    
    recommendations = data_clean.iloc[indices[0]]
    return recommendations[['title', 'clean_ingredients', 'image']]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ing = request.form['ing']
        
        # Check if input is valid
        if len(ing.strip()) < 3:
            return render_template(
                'index.html',
                recommendations=[],
                message="Please enter a valid ingredient list."
            )
        
        input_ingredients = ing  # User input ingredients
        recommendations = recommend_recipes(input_ingredients)  # Get recommended recipes
        
        if recommendations.empty:
            return render_template(
                'index.html',
                recommendations=[],
                message="No relevant recipes found for your search."
            )
        
        return render_template(
            'index.html',
            recommendations=recommendations.to_dict(orient='records')
        )
    return render_template('index.html', recommendations=[])

@app.route('/recipe/<title>')
def view_recipe(title):
    # Find the specific recipe by name
    recipe = data_clean[data_clean['title'] == title].to_dict(orient='records')
    
    if recipe:
        # Get the first matching recipe (in case of duplicates)
        recipe = recipe[0]
        
        # Clean the nutrition info
        if isinstance(recipe['nutrition_info'], str):
            # Remove percentage values and clean up the string
            nutrition = recipe['nutrition_info']
            nutrition = re.sub(r'\([^)]*\)', '', nutrition)  # Remove percentages in parentheses
            recipe['nutrition_info'] = nutrition
            
        # Clean the instructions if they're in string format
        if isinstance(recipe['instructions'], str):
            # Remove the outer brackets and quotes
            instructions = recipe['instructions'].strip('[]')
            # Split by various possible delimiters
            instructions = instructions.replace('", "', '|').replace('.,', '.|')
            # Clean up and store back
            recipe['instructions'] = instructions
            
        return render_template('recipe_detail.html', recipe=recipe)
    else:
        # Handle case where recipe is not found
        return "Recipe not found", 404


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)