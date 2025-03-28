from Chunking.fixed_token_chunker import FixedTokenChunker
import chromadb.utils.embedding_functions as embedding_functions
from Evaluation.single_corpus_evaluation import SingleCorpusEvaluation

# Retrieval Pipeline-File
def evaluate(n_results: int = 10, chunker: FixedTokenChunker = FixedTokenChunker(chunk_size=512, chunk_overlap=50), embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction('sentence-transformers/multi-qa-mpnet-base-dot-v1')):
    # Initialize evaluation with paths
    evaluation = SingleCorpusEvaluation(
        questions_csv_path="Data/questions_df.csv",
        corpus_path="Data/state_of_the_union.md"
    )

    # Run evaluation
    results = evaluation.evaluate_chunker(
        chunker=chunker,
        embedding_function=embedding_function,
        n_results=n_results
    )

    print(f"Precision: {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
    print(f"Recall: {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
    return results

def iterateNumResults():
    results = {}

    # Open file for writing
    with open("ComputedMetrics/num_results_experiment_1.txt", 'w') as f:
        # Iterate through different n_results values
        for n in range(1, 21):
            metrics = evaluate(n_results=n)

            # Format the results
            result_text = f"Number of results: {n}\n"
            result_text += f"Precision: {metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}\n"
            result_text += f"Recall: {metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}\n\n"

            # Store in results dictionary and write to file
            results[n] = result_text
            f.write(result_text)

    return results

def optimalChunkAndOverlap():
    chunk_sizes = [128, 256, 512, 1024]
    overlaps = [0, 50, 100, 200]
    grid_results = {}

    # Open file for writing
    with open("ComputedMetrics/optimal_chunk_overlap_experiment_2", 'w') as f:
        for size in chunk_sizes:
            for overlap in overlaps:
                if overlap >= size:  # Skip invalid combinations
                    continue

                chunker = FixedTokenChunker(chunk_size=size, chunk_overlap=overlap)
                metrics = evaluate(n_results=5, chunker=chunker)

                # Calculate F1 score
                p = metrics['precision_mean']
                r = metrics['recall_mean']
                f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

                # Format the results
                result_text = f"Chunk size: {size}, Overlap: {overlap}\n"
                result_text += f"Precision: {p:.4f} ± {metrics['precision_std']:.4f}\n"
                result_text += f"Recall: {r:.4f} ± {metrics['recall_std']:.4f}\n"
                result_text += f"F1 Score: {f1:.4f}\n\n"

                # Store in results dictionary and write to file
                grid_results[(size, overlap)] = f1
                f.write(result_text)

    return grid_results

def compareEmbeddingFunctions():
    import chromadb.utils.embedding_functions as embedding_functions
    embedding_functions = {
        "mpnet-base": embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2"),
        "MiniLM": embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"),
        "iniLM-L3-v2": embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-MiniLM-L3-v2"),
    }

    model_results = {}

    # Open file for writing
    with open("ComputedMetrics/compare_embeddings_experiment_3", 'w') as f:
        for name, emb_func in embedding_functions.items():
            metrics = evaluate(
                chunker=FixedTokenChunker(
                    chunk_size=512,
                    chunk_overlap=50
                ),  # Use a fixed chunker configuration
                embedding_function=emb_func,
                n_results=5
            )

            # Calculate F1
            p = metrics['precision_mean']
            r = metrics['recall_mean']
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

            # Format the results
            result_text = f"Embedding model: {name}\n"
            result_text += f"Precision: {p:.4f} ± {metrics['precision_std']:.4f}\n"
            result_text += f"Recall: {r:.4f} ± {metrics['recall_std']:.4f}\n"
            result_text += f"F1 Score: {f1:.4f}\n\n"

            # Store in results dictionary and write to file
            model_results[name] = {
                'precision': p,
                'recall': r,
                'f1': f1
            }
            f.write(result_text)

if __name__ == '__main__':
    # Experimentation Scripts
    iterateNumResults()
    optimalChunkAndOverlap()
    compareEmbeddingFunctions()

    #Sample single run
    evaluate(n_results=10, chunker=FixedTokenChunker(chunk_size=512, chunk_overlap=50))