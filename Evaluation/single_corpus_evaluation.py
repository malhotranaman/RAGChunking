import numpy as np
import pandas as pd
import json
import chromadb


class SingleCorpusEvaluation:
    def __init__(self, questions_csv_path, corpus_path):
        self.questions_csv_path = questions_csv_path
        self.corpus_path = corpus_path

        # Load questions and references
        self.questions_df = pd.read_csv(questions_csv_path)
        self.questions_df['references'] = self.questions_df['references'].apply(json.loads)

        # Load corpus content
        with open(corpus_path, 'r', encoding='utf-8') as file:
            self.corpus = file.read()

        self.chroma_client = chromadb.Client()

    def evaluate_chunker(self, chunker: object, embedding_function : object, n_results: int):
        """
        Evaluates chunker on single corpus.

        @Params
        chunker
            The text chunker used
        embedding_function : object
            The embedding function for retrieval
        n_results : int
            Num of chunks to retrieve per question
        """

        chunks = chunker.split_text(self.corpus)
        chunk_metadatas = []

        # Try search document position information for each doc chunk
        # @error for missing chunk in corpora
        for chunk in chunks:
            try:
                start_index = self.corpus.find(chunk)
                if start_index == -1:
                    raise ValueError(f"Chunk not found in corpus")
                end_index = start_index + len(chunk)
                chunk_metadatas.append({"start_index": start_index, "end_index": end_index})
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue

        # Chunk collection
        collection = self.chroma_client.create_collection("chunks", embedding_function=embedding_function)
        collection.add(
            documents=chunks,
            metadatas=chunk_metadatas,
            ids=[str(i) for i in range(len(chunks))]
        )

        # Start metric calculations
        precision_scores = []
        recall_scores = []

        for _, row in self.questions_df.iterrows():
            cur_question = row['question']
            cur_references = row['references']

            # Get chunks relevant for cur_question
            results = collection.query(
                query_texts=[cur_question],
                n_results=n_results
            )

            retrieved_chunks = results['metadatas'][0]

            # Calculation of metrics
            metrics = self._calculate_metrics(retrieved_chunks, cur_references)
            precision_scores.append(metrics['precision'])
            recall_scores.append(metrics['recall'])

        self.chroma_client.delete_collection("chunks")
        return {
            "precision_mean": np.mean(precision_scores),
            "precision_std": np.std(precision_scores),
            "recall_mean": np.mean(recall_scores),
            "recall_std": np.std(recall_scores)
        }

    def _calculate_metrics(self, retrieved_chunks, references):
        # Convert to ranges
        retrieved_ranges = [(chunk['start_index'], chunk['end_index']) for chunk in retrieved_chunks]
        reference_ranges = [(int(ref['start_index']), int(ref['end_index'])) for ref in references]

        # Intersections
        intersection_ranges = []
        for chunk_range in retrieved_ranges:
            for ref_range in reference_ranges:
                intersection = self._intersect_ranges(chunk_range, ref_range)
                if intersection:
                    intersection_ranges = self._union_ranges(intersection_ranges + [intersection])

        # Sizes
        intersection_size = self._sum_ranges(intersection_ranges)
        reference_size = self._sum_ranges(reference_ranges)
        retrieved_size = self._sum_ranges(retrieved_ranges)

        # Metrics corresponding to website formulas
        precision = intersection_size / retrieved_size if retrieved_size > 0 else 0
        recall = intersection_size / reference_size if reference_size > 0 else 0

        return {
            "precision": precision*100,
            "recall": recall*100
        }

    # Helper functions for range calculations - from website implementation
    def _sum_ranges(self, ranges):
        return sum(end - start for start, end in ranges)

    def _intersect_ranges(self, range1, range2):
        start1, end1 = range1
        start2, end2 = range2

        start = max(start1, start2)
        end = min(end1, end2)

        if start <= end:
            return start, end
        return None

    def _union_ranges(self, ranges):
        if not ranges:
            return []

        sorted_ranges = sorted(ranges, key=lambda x: x[0])

        merged = [sorted_ranges[0]]
        for current in sorted_ranges[1:]:
            previous = merged[-1]

            if current[0] <= previous[1]:
                merged[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged.append(current)

        return merged
