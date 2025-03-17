import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple
import argparse
import os
import requests
import json
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LLMValidator:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _call_api(self, payload):
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API Error: {str(e)}")
            return None
    
    def compare_recommendations(self, history_details, rec_set_a, rec_set_b):
        """Simulate user choosing between two recommendation sets (A/B)"""
        # Format movie lists
        watched_str = "\n".join([f"- {movie}" for movie in history_details])
        set_a_str = "\n".join([f"- {movie}" for movie in rec_set_a])
        set_b_str = "\n".join([f"- {movie}" for movie in rec_set_b])

        system_prompt = f"""You are a movie enthusiast who recently watched:
            {watched_str}

            You must choose between two recommendation sets. Consider:
            1. Which set better matches your tastes
            2. Which has more movies you'd actually watch
            Respond ONLY with 'A' or 'B' in lowercase."""

        user_prompt = f"""Set A:
            {set_a_str}

            Set B:
            {set_b_str}

            Which set would you choose? Answer:"""

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 2
        }

        response = self._call_api(payload)
        if not response:
            return None  # Handle error in calling code
            
        try:
            choice = response['choices'][0]['message']['content'].strip().lower()
            return 'a' if 'a' in choice else 'b' if 'b' in choice else None
        except:
            return None
    
    def simulate_user_response(self, history_details, recommendation_details):
        """Simulate a user's likelihood to watch recommendations (0-1 score)"""
        # Format movie lists
        watched_movies = "\n".join([f"- {movie}" for movie in history_details])
        recommended_movies = "\n".join([f"- {movie}" for movie in recommendation_details])

        # System prompt to simulate user behavior
        system_prompt = f"""You are a user who recently watched these movies:
            {watched_movies}
            
            You're being recommended new movies to watch. You will:
            1. Analyze your movie preferences based on your watch history
            2. Consider which recommended movies match your tastes
            3. Return a score between 0-1 (0 = would watch none, 1 = would watch all)
            Respond ONLY with the numerical score between 0 and 1 using 2 decimal places like 0.75."""

        # User prompt with recommendations
        user_prompt = f"""Recommended movies:
            {recommended_movies}
            
            What percentage of these recommendations would I actually watch? Convert this to a 0-1 score."""

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,  # More focused responses
            "max_tokens": 8
        }

        # Get and parse response
        response = self._call_api(payload)
        if not response:
            return 0.0
            
        try:
            score_text = response['choices'][0]['message']['content'].strip()
            score = float(re.search(r"0?\.\d{1,2}", score_text).group())
            return max(0.0, min(1.0, score))
        except:
            return 0.0

class CFBaseline:
    def __init__(self, factors=128, iterations=15, regularization=0.01):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.model = None
        self.sparse_matrix = None  # Store the sparse matrix for later use
        
    def fit(self, ratings):
        # Create sparse user-item matrix using correct column names.
        users = ratings['userId'].values
        items = ratings['movieId'].values
        values = np.ones(len(ratings))
        
        # Save the sparse matrix as an instance variable.
        self.sparse_matrix = csr_matrix(
            (values, (users, items)),
            shape=(users.max()+1, items.max()+1))
        
        # Weight using BM25 for better implicit feedback handling.
        weighted_matrix = bm25_weight(self.sparse_matrix, K1=100, B=0.8)
        
        # Train ALS model on the weighted matrix.
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            use_gpu=False  # Set to False if no GPU is available.
        )
        self.model.fit(weighted_matrix)
        
    def recommend(self, user_id, n=10):
        # Return recommendations for a given user.
        return self.model.recommend(
            userid=user_id,
            user_items=self.sparse_matrix[user_id],
            N=n,
            filter_already_liked_items=True
        )

# ----------------------------
# Data Loading & Preprocessing
# ----------------------------
class DataLoader:
    @staticmethod
    def load_movielens(data_path="ml-32m", subset_size=2000): 
        movies = pd.read_csv(f"{data_path}/movies.csv")
        ratings = pd.read_csv(f"{data_path}/ratings.csv")
        
        # Sample a representative subset (ensure genre diversity)
        if subset_size > 0:
            movies = movies.groupby('genres', group_keys=False).apply(
                lambda x: x.sample(min(len(x), subset_size//10), random_state=42)
            ).sample(frac=1).reset_index(drop=True)[:subset_size]
        else:
            movies = movies.sample(frac=1).reset_index(drop=True)
        
        # Create robust ID mapping
        id_mapping = {row.movieId: idx for idx, row in movies.iterrows()}
        subset_movie_ids = set(movies["movieId"])
        
        # Filter and map ratings
        ratings = ratings[ratings["movieId"].isin(subset_movie_ids)].copy()
        ratings['movieId'] = ratings['movieId'].map(id_mapping)
        ratings = ratings.dropna(subset=['movieId'])
        ratings['movieId'] = ratings['movieId'].astype(int)
        
        # Enhanced text representation
        movies["text"] = movies["title"] + " " + movies["genres"].str.replace('|', ' ') 
        print(f"Loaded {len(movies)} movies and {len(ratings)} ratings.")
        #print(f"Movies: {movies}")
        #print(f"Ratings: {ratings}")
        # Return movies, ratings, and the size of the dataset
        return movies, ratings, len(movies), len(ratings)

# ----------------------------
# Embedding & Clustering
# ----------------------------
class OnlineClustering:
    def __init__(self, initial_threshold=0.6, adjust_interval=150, dynamic=True, verbose=False):
        self.threshold = initial_threshold
        self.centroids = []
        self.cluster_items = defaultdict(list)
        self.item_embeddings = {}
        self.counter = 0
        self.adjust_interval = adjust_interval
        self.dynamic = dynamic
        self.verbose = verbose
        
    def add_item(self, embedding, item_idx):
        self.item_embeddings[item_idx] = embedding
        self.counter += 1

        # Original clustering logic
        if not self.centroids:
            self.centroids.append(embedding)
            self.cluster_items[0].append(item_idx)
            assigned_cluster = 0
        else:
            centroid_matrix = np.array(self.centroids)
            similarities = cosine_similarity([embedding], centroid_matrix)[0]
            similarities = similarities * (1 + np.random.uniform(-0.05, 0.05)) # Add noise to escape minimas
            max_sim_idx = np.argmax(similarities)

            if similarities[max_sim_idx] > self.threshold:
                self.cluster_items[max_sim_idx].append(item_idx)
                cluster_embeds = [self.item_embeddings[idx] for idx in self.cluster_items[max_sim_idx]]
                #self.centroids[max_sim_idx] = np.mean(cluster_embeds, axis=0)
                alpha = 0.05  # Weight for new item (adjust as needed)
                self.centroids[max_sim_idx] = (1 - alpha) * self.centroids[max_sim_idx] + alpha * embedding
                assigned_cluster = max_sim_idx
            else:
                new_idx = len(self.centroids)
                self.centroids.append(embedding)
                self.cluster_items[new_idx].append(item_idx)
                assigned_cluster = new_idx

        # Periodically adjust threshold after the new item has been assigned a cluster.
        if self.dynamic and (self.counter % self.adjust_interval == 0):
            if self.verbose:
                print(f"Adjusting threshold at {self.counter} items.")
            self._adjust_threshold()
            
        if self.verbose:
            print(f"Items: {len(self.item_embeddings)}. Clusters: {len(self.centroids)}. Threshold: {self.threshold:.2f}.")

        return assigned_cluster

    def _adjust_threshold(self):
        if len(self.centroids) < 2:
            print("Cannot adjust threshold with less than 2 clusters.")
            return  # Silhouette score requires at least 2 clusters
        
        # Prepare data for silhouette score
        item_ids = list(self.item_embeddings.keys())
        embeddings = np.array([self.item_embeddings[idx] for idx in item_ids])
        labels = []
        for idx in item_ids:
            for cluster, items in self.cluster_items.items():
                if idx in items:
                    labels.append(cluster)
                    break
        
        # Calculate silhouette score
        try:
            score = silhouette_score(embeddings, labels)
            if score < 0.15:  # Extremely poor clustering
                self.threshold = max(0.25, self.threshold * 0.95)
            elif score < 0.25:
                # Gentle decrease with higher floor
                self.threshold = max(0.25, self.threshold * 0.98)  
            elif score > 0.35:
                # Gentle increase
                self.threshold = min(0.65, self.threshold * 1.02)
            if self.verbose:
                print(f"Silhouette score: {score:.2f}") 
                print(f"New threshold: {self.threshold:.2f}")
        except Exception as e:
            print(f"Error calculating silhouette score: {e}")
            pass  # Handle cases with single-item clusters
        
    def add_item_old(self, embedding, item_idx):
        self.item_embeddings[item_idx] = embedding
        self.counter += 1
        
        # Periodically adjust threshold
        if self.dynamic and (self.counter % self.adjust_interval == 0):
            print(f"Adjusting threshold at {self.counter} items.")
            self._adjust_threshold()

        # Original clustering logic
        if not self.centroids:
            self.centroids.append(embedding)
            self.cluster_items[0].append(item_idx)
            return 0

        centroid_matrix = np.array(self.centroids)
        similarities = cosine_similarity([embedding], centroid_matrix)[0]
        max_sim_idx = np.argmax(similarities)

        if similarities[max_sim_idx] > self.threshold:
            self.cluster_items[max_sim_idx].append(item_idx)
            cluster_embeds = [self.item_embeddings[idx] for idx in self.cluster_items[max_sim_idx]]
            self.centroids[max_sim_idx] = np.mean(cluster_embeds, axis=0)
            return max_sim_idx
        else:
            new_idx = len(self.centroids)
            self.centroids.append(embedding)
            self.cluster_items[new_idx].append(item_idx)
            return new_idx

# ----------------------------
# Recommender System
# ----------------------------
class Recommender:
    def __init__(self, clustering_model, movies, embeddings, model):
        self.clustering = clustering_model
        self.movies = movies
        self.model = model #SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = embeddings

    def recommend_new_user(self, keywords, k=10):
        if not self.clustering.centroids:
            print("No clusters available. Generating random recommendations.")
            return self._sample_random(k)
        
        # Embed keywords directly
        keywords_embeddings = self.model.encode(keywords)
        query_embed = np.mean(keywords_embeddings, axis=0)
        
        centroid_matrix = np.array(self.clustering.centroids)
        cluster_sims = cosine_similarity([query_embed], centroid_matrix)[0]
        top_clusters = np.argsort(cluster_sims)[-3:] # Top 3 clusters
        
        return self._sample_from_clusters(top_clusters, k)
    
    def recommend_existing_user(self, history, k=10, explore=False):
        original_k = k  # store the original number of recommendations
        cluster_history = defaultdict(list)
        for item_id in history:
            # Assume item_id is a valid index
            cluster_id = None
            # Find which cluster contains the item
            for cluster, items in self.clustering.cluster_items.items():
                if item_id in items:
                    cluster_id = cluster
                    break
            if cluster_id is not None:
                cluster_history[cluster_id].append(item_id)
        
        sorted_clusters = sorted(cluster_history, key=lambda c: len(cluster_history[c]), reverse=True)
        top_clusters = sorted_clusters[:3]
        recommendations = []
        
        if explore:
            #all_clusters = list(self.clustering.cluster_items.keys())
            #non_top_clusters = list(set(all_clusters) - set(top_clusters))
            non_top_clusters = sorted_clusters[9:14]
            if non_top_clusters:
                print(f"Exploring non-top clusters")
                # Sample from non-top clusters (exploration)
                #recs = self._sample_from_clusters(non_top_clusters, max(original_k // 2, 1))
                #recs = self._sample_from_clusters(non_top_clusters, int(round((2/3)*original_k)))
                recs = self._sample_from_clusters(non_top_clusters, original_k)
                recommendations += recs
        
        # Determine how many recommendations are still needed
        needed = original_k - len(recommendations)
        # Sample from top clusters to fill the rest
        recs_top = self._sample_from_clusters(top_clusters, needed)
        recommendations += recs_top
        
        # Ensure exactly original_k items are returned
        return recommendations[:original_k]

    def recommend_existing_user_old(self, history, k=10, explore=False):
        cluster_history = defaultdict(list)
        for item_id in history:
            # Assume item_id is a valid index
            cluster_id = None
            # Find which cluster contains the item
            for cluster, items in self.clustering.cluster_items.items():
                if item_id in items:
                    cluster_id = cluster
                    break
            if cluster_id is not None:
                cluster_history[cluster_id].append(item_id)
        
        top_clusters = sorted(cluster_history, 
                            key=lambda c: len(cluster_history[c]), 
                            reverse=True)[:3]
        recommendations = []
        
        if explore:
            all_clusters = list(self.clustering.cluster_items.keys())
            non_top_clusters = list(set(all_clusters) - set(top_clusters))
            if non_top_clusters:
                recs = self._sample_from_clusters(non_top_clusters, max(k//2, 1))
                recommendations += recs
                k -= len(recs)
        
        # Add remaining from top clusters
        recs_top = self._sample_from_clusters(top_clusters, k)
        recommendations += recs_top
        
        return recommendations[:k]

    def _sample_from_clusters(self, clusters, k):
        sampled = []
        for cluster in clusters:
            items = self.clustering.cluster_items.get(cluster, [])
            if not items:
                continue
            n = max(1, k // len(clusters))
            sampled += list(np.random.choice(items, min(n, len(items)), replace=False))
        return sampled
    
    def _sample_random(self, k):
        return list(np.random.choice(len(self.movies), k, replace=False))

# ----------------------------
# Metrics
# ---------------------------- 
class Metrics:
    @staticmethod
    def ILS(recommendations, embeddings):
        if len(recommendations) < 2:
            return 0.0
        rec_embeds = embeddings[recommendations]
        sim_matrix = cosine_similarity(rec_embeds)
        np.fill_diagonal(sim_matrix, 0)
        return sim_matrix.sum() / (len(recommendations)*(len(recommendations)-1))
    
    @staticmethod
    def unexpectedness(recommendations, user_history, embeddings):
        if not recommendations or not user_history:
            return 0.0
        hist_embeds = embeddings[user_history]
        rec_embeds = embeddings[recommendations]
        return 1 - cosine_similarity(rec_embeds, hist_embeds).mean()

# ----------------------------
# Experiments
# ----------------------------
class Experiments:
    def __init__(self, movies, ratings, k=10, n_history_items=10, n_users=1000, verbose=False, api_key=None):
        self.api_key = api_key
        self.movies = movies
        self.ratings = ratings
        self.k = k
        self.n_history_items = n_history_items
        self.n_users = n_users
        self.model = SentenceTransformer('all-MiniLM-L12-v2')
        self.embeddings = self.model.encode(movies["text"].tolist())#, convert_to_tensor=True).cpu().numpy()
        self.embeddings = normalize(self.embeddings, axis=1) # L2-normalize
        self.verbose = verbose
        self.clustering = OnlineClustering(initial_threshold=0.45, adjust_interval=100, dynamic=True, verbose=self.verbose)
        for idx in range(len(self.movies)):
            self.clustering.add_item(self.embeddings[idx], idx)
            #print(f"Added item {idx} to clustering. Current clusters: {len(self.clustering.centroids)}. Threshold: {self.clustering.threshold:.2f}. Total items: {len(self.clustering.item_embeddings)}.")
        self.user_histories = self.build_realistic_histories()
        self.validator = LLMValidator(api_key=self.api_key)
        
    def build_realistic_histories(self):
        user_histories = []
        for user_id in self.ratings['userId'].unique():
            user_ratings = self.ratings[self.ratings['userId'] == user_id]
            if len(user_ratings) >= self.n_history_items:
                history = user_ratings.sample(self.n_history_items, random_state=42)['movieId'].tolist()
                if self.verbose:
                    print(f"User history: {history}")
                user_histories.append(history)
            if len(user_histories) == self.n_users:
                break
        print(f"User histories: {len(user_histories)}. Num of users: {self.n_users}.")
        return user_histories
        
    def build_realistic_histories_old(self):
        user_histories = []
        for user_id in self.ratings['userId'].unique():
            user_ratings = self.ratings[self.ratings['userId'] == user_id]
            print(f"User {user_id} has {len(user_ratings)} ratings.")
            if len(user_ratings) >= self.n_history_items:
                history = user_ratings.sample(self.n_history_items, random_state=42)['movieId'].tolist()
                print(f"User history: {history}")
                user_histories.append(history)
        print(f"User histories: {len(user_histories)}. Num of users: {self.n_users}.")
        return user_histories
        
    def run_cold_start(self):
        cold_start_recs = []
        recommender = Recommender(self.clustering, self.movies, self.embeddings, self.model)
        
        # Better keyword generation from movie texts
        all_keywords = set()
        for text in self.movies["text"]:
            all_keywords.update(text.lower().split())
        keywords_list = [np.random.choice(list(all_keywords), 3) for _ in range(self.n_users)]
        print(f"Total keywords (corresponds to n_users): {len(keywords_list)}. Example keyword: {keywords_list[0]}")
        
        # Compute ILS for each user and average over all users
        results = []
        for keywords in keywords_list:
            recs = recommender.recommend_new_user(keywords, k=self.k)
            cold_start_recs.append(recs)
            results.append(Metrics.ILS(recs, self.embeddings))
        
        return np.mean(results), cold_start_recs
    
    def run_exploration_mode(self):
        explore_on_recs=[]
        explore_off_recs=[]
        recommender = Recommender(self.clustering, self.movies, self.embeddings, self.model)
        
        # Group comparisons
        ils_off, unexp_off = [], []
        ils_on, unexp_on = [], []
        
        print("... Test Exploration mode: OFF")
        for h in self.user_histories:
            recs = recommender.recommend_existing_user(h, k=self.k, explore=False)
            explore_off_recs.append(recs)
            if self.verbose:
                print(f"Recommendations: {recs}")
            ils_off.append(Metrics.ILS(recs, self.embeddings))
            unexp_off.append(Metrics.unexpectedness(recs, h, self.embeddings))
        
        print("... Test Exploration mode: ON")
        for h in self.user_histories:
            recs = recommender.recommend_existing_user(h, k=self.k, explore=True)
            explore_on_recs.append(recs)
            if self.verbose:
                print(f"Recommendations: {recs}")
            ils_on.append(Metrics.ILS(recs, self.embeddings))
            unexp_on.append(Metrics.unexpectedness(recs, h, self.embeddings))
            
        print(f"... Size of ils_off: {len(ils_off)}, ils_on: {len(ils_on)}, unexp_off: {len(unexp_off)}, unexp_on: {len(unexp_on)}")
        
        return (np.mean(ils_off), np.mean(ils_on)), (np.mean(unexp_off), np.mean(unexp_on)), explore_off_recs, explore_on_recs
    
    def run_popularity_baseline(self):
        # Get popular movie indices from ratings (already remapped)
        popular_internal_ids = list(self.ratings['movieId'].value_counts().index[:self.k])
        print(f"Popular internal movie IDs: {popular_internal_ids}")

        # Build reverse mapping: internal index -> original movieId
        index_to_id = {idx: row.movieId for idx, row in self.movies.iterrows()}
        #if self.verbose:
            #print(f"Index to original movie ID mapping: {index_to_id}")

        # Map popular internal indices to their corresponding original movie IDs
        popular_original_ids = [index_to_id.get(movie_index) for movie_index in popular_internal_ids if movie_index in index_to_id]
        if self.verbose:
            print(f"Popular original movie IDs: {popular_original_ids}")

        # Ensure exactly k recommendations (both representations have the same length)
        popular_internal_ids = popular_internal_ids[:self.k]
        popular_original_ids = popular_original_ids[:self.k]

        if not popular_internal_ids:
            print("Warning: No valid popular indices found for baseline.")
            return 0.0, 0.0

        # Compute metrics using the internal indices (needed for accessing embeddings)
        unexp_scores = [
            Metrics.unexpectedness(popular_internal_ids, history, self.embeddings)
            for history in self.user_histories
        ]
        ils_score = Metrics.ILS(popular_internal_ids, self.embeddings)
        unexp_score = np.mean(unexp_scores) if unexp_scores else 0.0

        return ils_score, unexp_score

    def run_popularity_baseline_old(self):
        # Correct mapping from original movieId to internal indices
        popular_movie_ids = self.ratings['movieId'].value_counts().index[:self.k]
        print(f"Popular movie IDs: {popular_movie_ids}")
        
        id_to_index = {row.movieId: idx for idx, row in self.movies.iterrows()}
        print(f"ID to index: {id_to_index}")
        
        # Map movieIds to internal indices
        popular_indices = [id_to_index.get(movie_id) for movie_id in popular_movie_ids if movie_id in id_to_index]
        print(f"Popular indices: {popular_indices}")
        
        # Ensure exactly k recommendations
        popular_indices = popular_indices[:self.k]
        print(f"Popular indices: {popular_indices}")
        
        if not popular_indices:
            print("Warning: No valid popular indices found for baseline.")
            return 0.0, 0.0
        
        # Compute unexpectedness for each user
        unexp_scores = [
            Metrics.unexpectedness(popular_indices, history, self.embeddings)
            for history in self.user_histories
        ]
        
        ils_score = Metrics.ILS(popular_indices, self.embeddings)
        if unexp_scores:
            unexp_score = np.mean(unexp_scores)
        else:
            unexp_score = 0.0
        
        return ils_score, unexp_score

    def run_cf_baseline(self):
        cf_recs = []
        cf = CFBaseline(factors=64)
        cf.fit(self.ratings)
        
        # Sample test users using the correct column name
        test_users = np.random.choice(self.ratings['userId'].unique(), self.n_users)
        
        ils_scores = []
        unexp_scores = []
        
        for user, history in zip(test_users, self.user_histories):
            try:
                recs = cf.recommend(user, n=self.k)
                
                # Ensure recs is properly extracted if it's a tuple
                recs = recs[0] if isinstance(recs, tuple) else recs
                
                # Convert recs to a list if necessary
                recs = list(recs) if not isinstance(recs, list) else recs

                if len(recs) == self.k:
                    cf_recs.append(recs)
                    ils_score = Metrics.ILS(recs, self.embeddings)
                    ils_scores.append(ils_score)

                    unexp_score = Metrics.unexpectedness(recs, history, self.embeddings)
                    unexp_scores.append(unexp_score)
            except Exception as e:
                print(f"Error recommending for user {user}: {e}")
                continue
                    
        avg_ils = np.mean(ils_scores) if ils_scores else 0.0
        avg_unexp = np.mean(unexp_scores) if unexp_scores else 0.0
        
        return avg_ils, avg_unexp, cf_recs
    
    def _convert_to_original_ids(self, internal_indices: List[int]) -> List[int]:
        """Convert internal indices to original movie IDs"""
        index_to_id = {idx: row.movieId for idx, row in self.movies.iterrows()}
        return [index_to_id.get(idx) for idx in internal_indices if idx in index_to_id]
    
    def get_movie_details(self, movie_ids: List[int]) -> List[Dict]:
        """Get movie details (title, genres) from original movie IDs"""
        movie_details = []
        for movie_id in movie_ids:
            movie = self.movies[self.movies['movieId'] == movie_id].iloc[0]
            movie_details.append(movie['title'])
        return movie_details
    
    def LLM_validation(self, recs):
        scores = []
        for user_history, recommendations in zip(self.user_histories, recs):
            # Convert IDs to movie details
            history_details = self.get_movie_details(
                self._convert_to_original_ids(user_history)
            )
            recs_details = self.get_movie_details(
                self._convert_to_original_ids(recommendations)
            )
            
            # Get LLM score
            score = self.validator.simulate_user_response(
                history_details, 
                recs_details
            )
            scores.append(score)
            
            if self.verbose:
                print(f"Scored {score:.2f} for recommendations")

        # Report overall quality
        avg_score = sum(scores)/len(scores) if scores else 0
        print(f"\nAverage LLM Recommendation Score: {avg_score:.2f}/1.0")
        return scores
            
    def ab_test_recommendations(self, set_a, set_b):
        preferences = {'a': 0, 'b': 0, 'invalid': 0}
        
        # Verify equal lengths
        if not (len(self.user_histories) == len(set_a) == len(set_b)):
            raise ValueError("All input lists must have the same number of users")
        
        for user_history, recs_a, recs_b in zip(self.user_histories, set_a, set_b):
            # Convert IDs to movie details
            history_details = self.get_movie_details(
                self._convert_to_original_ids(user_history))
            set_a_details = self.get_movie_details(
                self._convert_to_original_ids(recs_a))
            set_b_details = self.get_movie_details(
                self._convert_to_original_ids(recs_b))

            # Get user choice
            choice = self.validator.compare_recommendations(
                history_details, 
                set_a_details,
                set_b_details
            )

            # Update counts
            if choice == 'a':
                preferences['a'] += 1
            elif choice == 'b':
                preferences['b'] += 1
            else:
                preferences['invalid'] += 1

            if self.verbose:
                print(f"User chose {choice.upper() if choice else 'Invalid'}")

        # Calculate percentages
        total_valid = preferences['a'] + preferences['b']
        preferences['total_users'] = len(self.user_histories)
        
        if total_valid > 0:
            preferences['a_pct'] = preferences['a'] / total_valid * 100
            preferences['b_pct'] = preferences['b'] / total_valid * 100
        else:
            preferences['a_pct'] = preferences['b_pct'] = 0.0
        
        return preferences
            
# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run recommender system experiments.")
    parser.add_argument('--n_users', type=int, default=1000, help='Number of users for experiments')
    parser.add_argument('--data_path', type=str, help='Path to the MovieLens dataset')
    parser.add_argument('--dataset_size', type=int, default=2000, help='Size of the dataset to load')
    parser.add_argument('--k', type=int, default=20, help='Number of recommendations per user')
    parser.add_argument('--n_history_items', type=int, default=10, help='Number of recommendations per user')
    parser.add_argument('--verbose', action='store_true', help='Print additional information')
    parser.add_argument('--deepseek_API_Key', type=str, help='API Key for DeepSeek')
    args = parser.parse_args()

    n_users = args.n_users
    subset_size = args.dataset_size
    k = args.k
    n_history_items = args.n_history_items
    verbose = args.verbose
    data_path = args.data_path
    api_key = args.deepseek_API_Key
    
    print("Loading data...")
    movies, ratings, dataset_size, rating_size = DataLoader.load_movielens(data_path=data_path, subset_size=subset_size)
    
    print("\nInitializing experiments...")
    exp = Experiments(movies, ratings, k=k, n_history_items=n_history_items, n_users=n_users, verbose=verbose, api_key=api_key)
    
    print("\nRunning cold start analysis...")
    cold_start_ils, cold_start_recs = exp.run_cold_start()
    
    print("\nTesting proposed method...")
    (ils_off, ils_on), (unexp_off, unexp_on), explore_off_recs, explore_on_recs = exp.run_exploration_mode()
    
    print("\nComputing pop baselines...")
    ils_pop, unexp_pop = exp.run_popularity_baseline()
    print("\nComputing CF baseline...")
    ils_cf, unexp_cf, cf_recs = exp.run_cf_baseline()
    
    print("\nRunning A/B Testing...")
    preferences = exp.ab_test_recommendations(explore_off_recs, explore_on_recs)
    
    """
    print("\nValidating cold start recs...")
    exp.LLM_validation(cold_start_recs)
    print("\nValidating exploration mode off recs...")
    exp.LLM_validation(explore_off_recs)
    print("\nValidating exploration mode on recs...")
    exp.LLM_validation(explore_on_recs)
    """
    
    print(f"""
    RESULTS with dataset size of {dataset_size}. Generating {k} recommendations per user.
    ---------------------------------------
    - Cold Start Performance:
    * ILS: {cold_start_ils:.2f} (lower is better)
    
    - Exploration Mode Off:
    * ILS: {ils_off:.2f}
    * Unexpectedness: {unexp_off:.2f}
 
    - Exploration Mode On:
    * ILS: {ils_on:.2f}
    * Unexpectedness: {unexp_on:.2f}
 
    - Baselines:
    * Collaborative Filtering - ILS: {ils_cf:.2f}, Unexpectedness: {unexp_cf:.2f}
    * Popularity - ILS: {ils_pop:.2f}, Unexpectedness: {unexp_pop:.2f}
    
    - A/B Testing:
    * Total Users: {preferences['total_users']}
    * Preference for Exploration Off: {preferences['a']} ({preferences['a_pct']:.1f}%)
    * Preference for Exploration On: {preferences['b']} ({preferences['b_pct']:.1f}%)
    * Invalid Responses: {preferences['invalid']}
    ---------------------------------------
    """)
