from rouge_score import rouge_scorer

def compute_rouge(generated_text: str, reference_text: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    return scores['rougeL'].fmeasure
