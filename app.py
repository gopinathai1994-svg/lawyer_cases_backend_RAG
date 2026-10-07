from flask import Flask, request, jsonify
from flask_cors import CORS

from evaluation import evaluate_text
from rag_engine import (
    ask_question,
    ingest_documents,
    list_documents
)


# ============================================================
# Flask App
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# Health Check
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "message": "Legal RAG API is running."
    })


# ============================================================
# List Documents
# ============================================================

@app.route("/documents", methods=["GET"])
def documents():
    try:

        documents = list_documents()

        return jsonify({
            "success": True,
            "documents": documents
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# Ingest Documents
# ============================================================

@app.route("/ingest", methods=["POST"])
def ingest():

    try:

        result = ingest_documents()

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# Chat
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json(
            silent=True
        )

        if not data:
            return jsonify({
                "success": False,
                "error": "JSON body is required."
            }), 400

        question = data.get(
            "question"
        )

        if not question:
            return jsonify({
                "success": False,
                "error": "question is required."
            }), 400

        question = str(
            question
        ).strip()

        if not question:
            return jsonify({
                "success": False,
                "error": "question cannot be empty."
            }), 400

        result = ask_question(
            question
        )

        return jsonify({
            "success": True,
            "question": question,
            "answer": result.get("answer"),
            "sources": result.get("sources", []),
            "retrieval_similarity": result.get(
                "retrieval_similarity",
                0.0
            ),
            "usage": result.get(
                "usage",
                {}
            )
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# Evaluate One Question
# ============================================================

@app.route("/evaluate", methods=["POST"])
def evaluate():

    try:

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({
                "success": False,
                "error": "JSON body is required."
            }), 400

        question = data.get(
            "question"
        )

        expected_answer = data.get(
            "expected_answer"
        )

        key_facts = data.get(
            "key_facts",
            []
        )

        # ----------------------------------------------------
        # Validate question
        # ----------------------------------------------------

        if not question:

            return jsonify({
                "success": False,
                "error": "question is required."
            }), 400

        # ----------------------------------------------------
        # Validate expected answer
        # ----------------------------------------------------

        if not expected_answer:

            return jsonify({
                "success": False,
                "error": (
                    "expected_answer is required for "
                    "BERTScore, ROUGE and METEOR evaluation."
                )
            }), 400

        # ----------------------------------------------------
        # Validate key facts
        # ----------------------------------------------------

        if key_facts is None:
            key_facts = []

        if not isinstance(
            key_facts,
            list
        ):

            return jsonify({
                "success": False,
                "error": "key_facts must be an array."
            }), 400

        # ----------------------------------------------------
        # Ask RAG
        # ----------------------------------------------------

        rag_result = ask_question(
            question
        )

        generated_answer = rag_result.get(
            "answer",
            ""
        )

        retrieval_similarity = rag_result.get(
            "retrieval_similarity",
            0.0
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        metrics = evaluate_text(

            reference_answer=expected_answer,

            generated_answer=generated_answer,

            retrieval_similarity=retrieval_similarity,

            key_facts=key_facts
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "question": question,

            "expected_answer": expected_answer,

            "generated_answer": generated_answer,

            "key_facts": key_facts,

            "evaluation": metrics,

            "sources": rag_result.get(
                "sources",
                []
            ),

            "usage": rag_result.get(
                "usage",
                {}
            )
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# Evaluate All Test Cases
# ============================================================

@app.route("/evaluate-all", methods=["POST"])
def evaluate_all():

    try:

        import json
        import os

        # ----------------------------------------------------
        # Test case file path
        # ----------------------------------------------------

        test_file = os.path.join(
            "evaluation",
            "test_cases.json"
        )

        # ----------------------------------------------------
        # Check file
        # ----------------------------------------------------

        if not os.path.exists(
            test_file
        ):

            return jsonify({
                "success": False,
                "error": (
                    f"Test case file not found: "
                    f"{test_file}"
                )
            }), 404

        # ----------------------------------------------------
        # Read test cases
        # ----------------------------------------------------

        with open(
            test_file,
            "r",
            encoding="utf-8"
        ) as file:

            test_cases = json.load(
                file
            )

        # ----------------------------------------------------
        # Validate test cases
        # ----------------------------------------------------

        if not isinstance(
            test_cases,
            list
        ):

            return jsonify({
                "success": False,
                "error": (
                    "test_cases.json must contain "
                    "an array."
                )
            }), 400

        # ----------------------------------------------------
        # Evaluation results
        # ----------------------------------------------------

        results = []

        total_score = 0.0

        successful_cases = 0

        failed_cases = 0

        # ----------------------------------------------------
        # Process every test case
        # ----------------------------------------------------

        for case in test_cases:

            try:

                # --------------------------------------------
                # Validate case
                # --------------------------------------------

                if not isinstance(
                    case,
                    dict
                ):

                    raise ValueError(
                        "Each test case must be an object."
                    )

                case_id = case.get(
                    "id"
                )

                question = case.get(
                    "question"
                )

                expected_answer = case.get(
                    "expected_answer"
                )

                key_facts = case.get(
                    "key_facts",
                    []
                )

                # --------------------------------------------
                # Required fields
                # --------------------------------------------

                if not question:

                    raise ValueError(
                        "question is missing."
                    )

                if not expected_answer:

                    raise ValueError(
                        "expected_answer is missing."
                    )

                if not isinstance(
                    key_facts,
                    list
                ):

                    raise ValueError(
                        "key_facts must be an array."
                    )

                # --------------------------------------------
                # Ask RAG
                # --------------------------------------------

                rag_result = ask_question(
                    question
                )

                generated_answer = rag_result.get(
                    "answer",
                    ""
                )

                retrieval_similarity = rag_result.get(
                    "retrieval_similarity",
                    0.0
                )

                # --------------------------------------------
                # Evaluate
                # --------------------------------------------

                evaluation = evaluate_text(

                    reference_answer=expected_answer,

                    generated_answer=generated_answer,

                    retrieval_similarity=retrieval_similarity,

                    key_facts=key_facts
                )

                # --------------------------------------------
                # Store result
                # --------------------------------------------

                case_result = {

                    "id": case_id,

                    "question": question,

                    "expected_answer": expected_answer,

                    "generated_answer": generated_answer,

                    "key_facts": key_facts,

                    "evaluation": evaluation,

                    "sources": rag_result.get(
                        "sources",
                        []
                    ),

                    "usage": rag_result.get(
                        "usage",
                        {}
                    ),

                    "success": True
                }

                results.append(
                    case_result
                )

                # --------------------------------------------
                # Summary calculation
                # --------------------------------------------

                total_score += evaluation[
                    "overall_score"
                ]

                successful_cases += 1

            except Exception as case_error:

                failed_cases += 1

                results.append({

                    "id": case.get(
                        "id"
                    ) if isinstance(
                        case,
                        dict
                    ) else None,

                    "question": case.get(
                        "question"
                    ) if isinstance(
                        case,
                        dict
                    ) else None,

                    "success": False,

                    "error": str(
                        case_error
                    )
                })

        # ----------------------------------------------------
        # Average score
        # ----------------------------------------------------

        if successful_cases > 0:

            average_score = (
                total_score
                / successful_cases
            )

        else:

            average_score = 0.0

        # ----------------------------------------------------
        # Overall quality
        # ----------------------------------------------------

        if average_score >= 0.90:

            overall_quality = "EXCELLENT"

        elif average_score >= 0.75:

            overall_quality = "GOOD"

        elif average_score >= 0.55:

            overall_quality = "MEDIUM"

        else:

            overall_quality = "LOW"

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "summary": {

                "total_cases": len(
                    test_cases
                ),

                "successful_cases": (
                    successful_cases
                ),

                "failed_cases": (
                    failed_cases
                ),

                "average_score": round(
                    average_score,
                    4
                ),

                "average_percentage": round(
                    average_score * 100,
                    2
                ),

                "quality": overall_quality
            },

            "results": results
        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# Run Flask Application
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )