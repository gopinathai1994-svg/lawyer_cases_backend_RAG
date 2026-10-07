from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from config import BERTSCORE_LANG


# ============================================================
# Utility
# ============================================================

def clamp01(value):
    """
    Make sure the score is always between 0 and 1.
    """
    return max(0.0, min(1.0, float(value)))


# ============================================================
# BERTScore
# ============================================================

def calculate_bertscore(reference_answer, generated_answer):
    """
    Calculate semantic similarity between
    reference answer and generated answer.
    """

    _, _, bert_f1 = bert_score(
        [generated_answer],
        [reference_answer],
        lang=BERTSCORE_LANG,
        rescale_with_baseline=True,
    )

    return clamp01(bert_f1.item())


# ============================================================
# ROUGE
# ============================================================

def calculate_rouge(reference_answer, generated_answer):
    """
    Calculate ROUGE-1 and ROUGE-L.
    """

    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"],
        use_stemmer=True
    )

    result = rouge.score(
        reference_answer,
        generated_answer
    )

    rouge1 = clamp01(
        result["rouge1"].fmeasure
    )

    rougeL = clamp01(
        result["rougeL"].fmeasure
    )

    return rouge1, rougeL


# ============================================================
# METEOR
# ============================================================

def calculate_meteor(reference_answer, generated_answer):
    """
    Calculate METEOR score.
    """

    reference_tokens = reference_answer.split()
    generated_tokens = generated_answer.split()

    score = meteor_score(
        [reference_tokens],
        generated_tokens
    )

    return clamp01(score)


# ============================================================
# KEY FACT COVERAGE
# ============================================================

def calculate_key_fact_coverage(
    generated_answer,
    key_facts
):
    """
    Check how many important facts are covered
    by the generated answer.

    Example:

    key_facts = [
        "The Act was enacted in 1988.",
        "The Act deals with corruption.",
        "The Act concerns public servants."
    ]

    If generated answer covers 2 out of 3 facts:

        coverage = 2 / 3
                 = 0.66
    """

    if not key_facts:
        return 0.0

    if not generated_answer:
        return 0.0

    generated = generated_answer.lower()

    matched_facts = 0

    for fact in key_facts:

        if not fact:
            continue

        fact_words = fact.lower().split()

        # Remove very small/common words
        important_words = [
            word.strip(".,!?()[]{}:;\"'")
            for word in fact_words
            if len(word.strip(".,!?()[]{}:;'\"")) > 3
        ]

        if not important_words:
            continue

        matched_words = 0

        for word in important_words:

            if word in generated:
                matched_words += 1

        fact_coverage = (
            matched_words / len(important_words)
        )

        # Consider the fact covered
        # when at least 50% of important words appear.
        if fact_coverage >= 0.50:
            matched_facts += 1

    return clamp01(
        matched_facts / len(key_facts)
    )


# ============================================================
# OVERALL SCORE
# ============================================================

def calculate_overall_score(
    bert_score_f1,
    rouge_l_f1,
    meteor,
    retrieval_similarity,
    key_fact_coverage
):
    """
    Calculate final evaluation score.

    Current weights:

        BERTScore             = 10%
        ROUGE-L               = 10%
        METEOR                = 5%
        Retrieval Similarity  = 25%
        Key Fact Coverage     = 50%

    Total = 100%
    """

    score = (
        (bert_score_f1 * 0.10)
        + (rouge_l_f1 * 0.10)
        + (meteor * 0.05)
        + (retrieval_similarity * 0.25)
        + (key_fact_coverage * 0.50)
    )

    return clamp01(score)


# ============================================================
# QUALITY LABEL
# ============================================================

def get_quality(score):
    """
    Convert numerical score into a readable quality label.
    """

    if score >= 0.90:
        return "EXCELLENT"

    elif score >= 0.75:
        return "GOOD"

    elif score >= 0.55:
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def evaluate_text(
    reference_answer,
    generated_answer,
    retrieval_similarity,
    key_facts=None
):
    """
    Run all evaluation metrics.

    Required:

        reference_answer
        generated_answer
        retrieval_similarity

    Optional:

        key_facts
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not reference_answer:
        raise ValueError(
            "reference_answer is required for "
            "BERTScore, ROUGE and METEOR evaluation."
        )

    if not generated_answer:
        raise ValueError(
            "generated_answer is required for evaluation."
        )

    reference_answer = str(
        reference_answer
    ).strip()

    generated_answer = str(
        generated_answer
    ).strip()

    if not reference_answer:
        raise ValueError(
            "reference_answer cannot be empty."
        )

    if not generated_answer:
        raise ValueError(
            "generated_answer cannot be empty."
        )

    # --------------------------------------------------------
    # Retrieval similarity
    # --------------------------------------------------------

    retrieval_similarity = clamp01(
        retrieval_similarity
    )

    # --------------------------------------------------------
    # BERTScore
    # --------------------------------------------------------

    bert_score_f1 = calculate_bertscore(
        reference_answer,
        generated_answer
    )

    # --------------------------------------------------------
    # ROUGE
    # --------------------------------------------------------

    rouge1_f1, rouge_l_f1 = calculate_rouge(
        reference_answer,
        generated_answer
    )

    # --------------------------------------------------------
    # METEOR
    # --------------------------------------------------------

    meteor = calculate_meteor(
        reference_answer,
        generated_answer
    )

    # --------------------------------------------------------
    # Key Fact Coverage
    # --------------------------------------------------------

    key_facts = key_facts or []

    key_fact_coverage = calculate_key_fact_coverage(
        generated_answer,
        key_facts
    )

    # --------------------------------------------------------
    # Overall Score
    # --------------------------------------------------------

    overall_score = calculate_overall_score(
        bert_score_f1=bert_score_f1,
        rouge_l_f1=rouge_l_f1,
        meteor=meteor,
        retrieval_similarity=retrieval_similarity,
        key_fact_coverage=key_fact_coverage
    )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    quality = get_quality(
        overall_score
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "metrics": {
            "bert_score_f1": round(
                bert_score_f1,
                4
            ),

            "rouge1_f1": round(
                rouge1_f1,
                4
            ),

            "rougeL_f1": round(
                rouge_l_f1,
                4
            ),

            "meteor": round(
                meteor,
                4
            ),

            "retrieval_similarity": round(
                retrieval_similarity,
                4
            ),

            "key_fact_coverage": round(
                key_fact_coverage,
                4
            )
        },

        "weights": {
            "bert_score_f1": 0.10,
            "rougeL_f1": 0.10,
            "meteor": 0.05,
            "retrieval_similarity": 0.25,
            "key_fact_coverage": 0.50
        },

        "overall_score": round(
            overall_score,
            4
        ),

        "percentage": round(
            overall_score * 100,
            2
        ),

        "quality": quality,

        "key_facts_checked": len(
            key_facts
        )
    }