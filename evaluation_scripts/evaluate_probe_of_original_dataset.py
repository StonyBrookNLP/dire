"""
Script to evaluate predictions on the probe of the original dataset (HotpotQA).
"""

from collections import defaultdict
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from metrics import (
    BaseMetric,
    ProbeAnswerMetric,
    ProbeSupportingFactsMetric
)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to evaluate predictions on"
                                     " the probe of the original dataset (HotpotQA).")
    parser.add_argument('probe_input_file_path', type=str)
    parser.add_argument('probe_prediction_file_path', type=str)
    parser.add_argument('original_prediction_file_path', type=str)

    args = parser.parse_args()

    with open(args.probe_input_file_path, "r") as file:
        probe_inputs = json.load(file)

    with open(args.probe_prediction_file_path, "r") as file:
        probe_predictions = [json.loads(line)
                             for line in file.readlines() if line.strip()]

    with open(args.original_prediction_file_path, "r") as file:
        original_predictions = [json.loads(line)
                                for line in file.readlines() if line.strip()]

    answer_metric = ProbeAnswerMetric()
    para_support_metric = ProbeSupportingFactsMetric()
    sent_support_metric = ProbeSupportingFactsMetric()

    probe_predictions_ = defaultdict(list)
    for prediction in probe_predictions:
        probe_predictions_[prediction["question_id"]].append(prediction)
    probe_predictions = probe_predictions_

    question_ids = set()
    for input_instance in probe_inputs:
        question_id = input_instance["_id"]
        question_ids.add(question_id)

        label_answer = input_instance["answer"]
        label_supporting_sentences = input_instance["supporting_facts"]
        label_supporting_paragraphs = set([info[0]
                                           for info in label_supporting_sentences])

        probe_prediction = probe_predictions[question_id].pop()
        predicted_answer = probe_prediction["answer"]
        predicted_confidence = probe_prediction["answer_confidence"]
        predicted_supporting_paragraphs = probe_prediction["supporting_paragraphs"]
        predicted_supporting_sentences = probe_prediction["supporting_sentences"]

        answer_metric.store_prediction(
            predicted_answer=predicted_answer,
            predicted_confidence=predicted_confidence,
            label_answer=label_answer,
            question_id=question_id,
            is_probe=True
        )

        para_support_metric.store_prediction(
            predicted_supporting_facts=predicted_supporting_paragraphs,
            label_supporting_facts=label_supporting_paragraphs,
            question_id=question_id,
            is_probe=True
        )

        sent_support_metric.store_prediction(
            predicted_supporting_facts=predicted_supporting_sentences,
            label_supporting_facts=label_supporting_sentences,
            question_id=question_id,
            is_probe=True
        )

    original_predictions = {prediction["question_id"]: prediction
                            for prediction in original_predictions}
    for question_id in question_ids:

        original_prediction = original_predictions[question_id]
        predicted_answer = original_prediction["answer"]
        predicted_supporting_paragraphs = original_prediction["supporting_paragraphs"]
        predicted_supporting_sentences = original_prediction["supporting_sentences"]

        answer_metric.store_prediction(
            predicted_answer=predicted_answer,
            question_id=question_id,
            is_probe=False
        )

        para_support_metric.store_prediction(
            predicted_supporting_facts=predicted_supporting_paragraphs,
            question_id=question_id,
            is_probe=False
        )

        sent_support_metric.store_prediction(
            predicted_supporting_facts=predicted_supporting_sentences,
            question_id=question_id,
            is_probe=False
        )

    answer_scores = answer_metric.compute_dataset_scores()
    supporting_para_scores = para_support_metric.compute_dataset_scores()
    supporting_sent_scores = sent_support_metric.compute_dataset_scores()

    joint_answer_supporting_para_scores = BaseMetric.compute_joint_dataset_scores(
        answer_metric,
        para_support_metric
    )

    joint_answer_supporting_sent_scores = BaseMetric.compute_joint_dataset_scores(
        answer_metric,
        sent_support_metric
    )

    print(f"Answer Scores")
    print(json.dumps(answer_scores, indent=4))

    print(f"Supporting Paragraph Scores")
    print(json.dumps(supporting_para_scores, indent=4))

    print(f"Supporting Sentence Scores")
    print(json.dumps(supporting_sent_scores, indent=4))

    print(f"Joint Answer + Supporting Paragraph Scores")
    print(json.dumps(joint_answer_supporting_para_scores, indent=4))

    print(f"Joint Answer + Supporting Sentence Scores")
    print(json.dumps(joint_answer_supporting_sent_scores, indent=4))
