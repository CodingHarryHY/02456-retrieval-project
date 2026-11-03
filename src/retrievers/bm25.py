from scipy.sparse import lil_matrix
from src.retrievers.tf_idf import TFIDF
from src.data_utils.document import Document
import numpy as np

class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b
        self.bm25_matrix = self.GetDocumentsBM25Vectors()
    
    def GetDocumentLengths(self):
            document_lengths = {}
            average_document_length = 0
            for document in self.index.GetDocuments():
                length = len(self.PreprocessText(document.GetText()))
                document_lengths[document] = length
                average_document_length += length
            return document_lengths, average_document_length/len(self.index.GetDocuments())    
        
    # Add methods for computing self.bm25_matrix
    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """Compute IDF using the formula specified in the project."""
        num_documents = len(self.index.GetDocuments())
        document_frequencies = self.GetDocumentFrequencies()

        idf = {}
        for term, freq in document_frequencies.items():
            idf[term] = np.log((num_documents + 1) / (freq + 0.5))

        return idf

    def GetDocumentsBM25Vectors(self):
        """build a sparse matrix of BM25 weights"""
        documents = self.index.GetDocuments()

        # term-to-column mapping for BIM25
        self.term_to_index={ term: index for index, term in enumerate(self.corpus_vocabulary) }

        bm25_matrix = lil_matrix((len(documents), len(self.corpus_vocabulary)), dtype=np.float32)
        if not documents:
            return bm25_matrix.tocsr()
        document_lengths, average_document_length = self.GetDocumentLengths()

        for row_index, document in enumerate(documents):
            term_counts = self.GetDocumentTermCounts(document)

            #empty document remains zero row
            if not term_counts:
                continue

            document_length = document_lengths[document]
            length_normalization = (1 - self.b) + self.b * document_length / average_document_length

            for term, count in term_counts.items():
                column_index = self.term_to_index[term]

                # BM25 applies saturation directly to raw term counts
                saturated_ft = (count * (self.k1 + 1)) / (count + self.k1 * length_normalization)

                bm25_matrix[row_index, column_index] = saturated_ft * self.idf[term]

        return bm25_matrix.tocsr()


    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        return scores