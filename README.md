# Cluster-Driven Recommender System

This project implements a cluster-driven recommender system for movies using collaborative filtering, content-based filtering, and clustering techniques. The system is designed to provide personalized movie recommendations to users based on their watch history and preferences.

## Features

- **Collaborative Filtering**: Uses Alternating Least Squares (ALS) for matrix factorization.
- **Content-Based Filtering**: Utilizes Sentence Transformers for movie embeddings.
- **Online Clustering**: Dynamically clusters movies based on their embeddings.
- **Exploration Mode**: Provides recommendations with an option to explore less popular clusters.
- **Cold Start Handling**: Generates recommendations for new users based on keywords.
- **Metrics**: Computes Intra-List Similarity (ILS) and unexpectedness for evaluation.
- **LLM Validation**: Simulates user responses using a language model to validate recommendations.

## Usage

### Data Preparation

1. Download the MovieLens dataset.

### Running the Recommender System

1. Run the recommender system experiments:
    ```sh
    python recommender.py --data_path data/ml-32m --n_users 1000 --dataset_size 2000 --k 20 --n_history_items 10 --verbose --deepseek_API_Key YOUR_API_KEY
    ```

### Arguments

- `--n_users`: Number of users for experiments (default: 1000)
- `--data_path`: Path to the MovieLens dataset
- `--dataset_size`: Size of the dataset to load (default: 2000)
- `--k`: Number of recommendations per user (default: 20)
- `--n_history_items`: Number of history items per user (default: 10)
- `--verbose`: Print additional information
- `--deepseek_API_Key`: API Key for DeepSeek

## Results

The results of the experiments will be printed to the console, including metrics such as ILS, unexpectedness, and user preferences from A/B testing.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
