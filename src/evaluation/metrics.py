from src.data_utils.query import Query

def aggregateConfusionMatrix(relevancies: list[list[bool]],queries: list[Query],k: int,num_documents: int):
    """Aggregate confusion matrix for retrieval outcomes at cutoff k within a fixed corpus"""
    totals = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}


    for query_relevancies, query in zip(relevancies, queries):
        counts = calculateConfusionMatrix(
            query_relevancies, query, k, num_documents
        )
        for name in totals:
            totals[name] += counts[name]

    return totals

def calculatePrecision(relevancies: list[bool], k: int):
    if k <= 0:
        raise ValueError("k must > 0.")

    return sum(relevancies[:k]) / k

def calculateRecall(relevancies: list[bool], query: Query):
    total_relevant = query.getNumberOfRelevantDocuments()
    if total_relevant == 0:
        return 0.0
    return sum(relevancies) / total_relevant

def calculateReciprocalRank(relevancies: list[bool]):
    for rank, is_relevant in enumerate(relevancies, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def Evaluate(relevancies: list[list[bool]],queries: list[Query],k: int):
    if k <= 0:
        raise ValueError("k must > 0.")
    if len(relevancies) != len(queries):
        raise ValueError("Each query must have 1 relevance list.")
    if not queries:
        raise ValueError("Cannot evaluate an empty query set.")

    total_precision = 0.0
    total_recall = 0.0
    total_rr = 0.0

    for query_relevancies, query in zip(relevancies, queries):
        top_k = query_relevancies[:k]

        total_precision += calculatePrecision(top_k, k)
        total_recall += calculateRecall(top_k, query)
        total_rr += calculateReciprocalRank(top_k)

    num_queries = len(queries)

    return {
        f"precision@{k}": total_precision / num_queries,
        f"recall@{k}": total_recall / num_queries,
        f"mrr@{k}": total_rr / num_queries
    }

def calculateConfusionMatrix(relevancies: list[bool], query: Query, k: int, num_documents: int):
    """Add a confusion matrix for retrieval outcomes at cutoff k within a fixed corpus"""
    if k <= 0:
        raise ValueError("k must be positive.")

    top_k = relevancies[:k]

    tp = sum(top_k)
    fp = len(top_k) - tp
    fn = query.getNumberOfRelevantDocuments() - tp

    tn = num_documents - tp - fp - fn

    if min(tp, fp, fn, tn) < 0:
        raise ValueError("Relevance counts and corpus size are inconsistent.")

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}