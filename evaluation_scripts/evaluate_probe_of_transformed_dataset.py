"""
Script to evaluate predictions on the probe of the transformed dataset (HotpotQA).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from metrics import (
    BaseMetric,
    ProbeTransformedAnswerMetric,
    ProbeTransformedSupportingFactsMetric
)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Script to evaluate predictions"
                                     " on the probe of the transformed dataset (HotpotQA).")
    parser.add_argument('input_file_path', type=str)
    parser.add_argument('prediction_file_path', type=str)

    args = parser.parse_args()

    with open(args.input_file_path, "r") as file:
        inputs = json.load(file)

    with open(args.prediction_file_path, "r") as file:
        predictions = [json.loads(line) for line in file.readlines() if line.strip()]

    answer_metric = ProbeTransformedAnswerMetric()
    para_support_metric = ProbeTransformedSupportingFactsMetric()
    sent_support_metric = ProbeTransformedSupportingFactsMetric()

    ids2key = lambda qid, sub_idx: "__".join([qid, str(sub_idx)])
    predictions = {ids2key(prediction["question_id"], prediction["sub_idx"]):
                   prediction for prediction in predictions}
    for input_instance in inputs:
        question_id = input_instance["_id"]
        sub_idx = input_instance["sub_idx"]
        instance_key = ids2key(question_id, sub_idx)


        label_answer = input_instance["answer"]
        label_supporting_sentences = input_instance["supporting_facts"]
        label_supporting_paragraphs = set([info[0]
                                           for info in label_supporting_sentences])
        label_sufficiency = input_instance["sufficiency"]

        prediction = predictions[instance_key]
        predicted_answer = prediction["answer"]
        predicted_confidence = prediction["answer_confidence"]
        predicted_supporting_paragraphs = prediction["supporting_paragraphs"]
        predicted_supporting_sentences = prediction["supporting_sentences"]
        predicted_sufficiency = prediction["sufficiency"]

        answer_metric.store_prediction(
            predicted_answer=predicted_answer,
            predicted_confidence=predicted_confidence,
            label_answer=label_answer,
            label_sufficiency=label_sufficiency,
            question_id=question_id,
            predicted_sufficiency=predicted_sufficiency
        )

        para_support_metric.store_prediction(
            predicted_supporting_facts=predicted_supporting_paragraphs,
            label_supporting_facts=label_supporting_paragraphs,
            label_sufficiency=label_sufficiency,
            question_id=question_id,
            predicted_sufficiency=predicted_sufficiency
        )

        sent_support_metric.store_prediction(
            predicted_supporting_facts=predicted_supporting_sentences,
            label_supporting_facts=label_supporting_sentences,
            label_sufficiency=label_sufficiency,
            question_id=question_id,
            predicted_sufficiency=predicted_sufficiency
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
