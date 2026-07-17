import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item interaction data
data = {
    'User': ['Alice', 'Alice', 'Bob', 'Bob', 'Carol', 'Carol', 'Dave', 'Dave', 'Eve', 'Eve'],
    'Item': ['Item1', 'Item2', 'Item2', 'Item3', 'Item1', 'Item3', 'Item2', 'Item4', 'Item1', 'Item4'],
    'Rating': [5, 3, 4, 2, 4, 5, 3, 4, 2, 5]
}

df = pd.DataFrame(data)

# Create user-item matrix
user_item_matrix = df.pivot_table(index='User', columns='Item', values='Rating')

# Normalize ratings by subtracting user mean
user_ratings_mean = user_item_matrix.mean(axis=1)
user_ratings_diff = user_item_matrix.sub(user_ratings_mean, axis=0).fillna(0)

# Compute user similarity matrix
user_similarity = pd.DataFrame(cosine_similarity(user_ratings_diff),
                               index=user_ratings_diff.index,
                               columns=user_ratings_diff.index)

def get_user_based_recommendations(target_user, top_n=3):
    # Find similar users
    sim_scores = user_similarity[target_user].drop(index=target_user)
    sim_scores = sim_scores / sim_scores.sum()  # Normalize to weights
    # Weighted sum of ratings from similar users
    weighted_ratings = user_ratings_diff.T.dot(sim_scores)
    # Add back user mean ratings
    user_mean = user_ratings_mean[target_user]
    recommendations = weighted_ratings / sim_scores.sum() + user_mean

    # Exclude items already rated
    rated_items = user_item_matrix.loc[target_user].dropna().index
    recommendations = recommendations[~recommendations.index.isin(rated_items)]
    # Return top recommendations
    return recommendations.sort_values(ascending=False).head(top_n)

def get_item_based_recommendations(target_user, top_n=3):
    # Compute item similarity matrix
    item_ratings = user_item_matrix.fillna(0).T
    item_similarity = pd.DataFrame(cosine_similarity(item_ratings),
                                   index=item_ratings.index,
                                   columns=item_ratings.index)

    # Get items rated by user
    user_ratings = user_item_matrix.loc[target_user].dropna()
    scores = pd.Series(dtype='float64')

    for item, rating in user_ratings.items():
        # Similar items to this one
        sim_scores = item_similarity[item]
        # Accumulate scores weighted by user's rating
        scores = scores.add(sim_scores * rating, fill_value=0)

    # Exclude items already rated
    scores = scores[~scores.index.isin(user_ratings.index)]
    return scores.sort_values(ascending=False).head(top_n)

# Example usage:
target_user = 'Alice'
print(f"\nUser-Based Recommendations for {target_user}:\n{get_user_based_recommendations(target_user)}")
print(f"\nItem-Based Recommendations for {target_user}:\n{get_item_based_recommendations(target_user)}")