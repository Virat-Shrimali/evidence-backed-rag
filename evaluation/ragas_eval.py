"""Generation evaluation metrics using Ragas / custom LLM-as-a-judge."""



def evaluate_generation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """Evaluate faithfulness, answer relevancy, and correctness."""
    # Scaffold interface for Ragas integration
    return {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "answer_correctness": 0.0,
    }
