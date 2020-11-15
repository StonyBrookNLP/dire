from typing import List, Dict
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy

from metrics import hotpotqa_eval, BaseMetric


@dataclass
class LabelPredictionInstance:
    label_answer: str = None

    probe_predicted_answers: List[str] = field(default_factory=lambda: deepcopy([]))
    probe_predicted_confidences: List[float] = field(default_factory=lambda: deepcopy([]))

    original_predicted_answer: str = None


class ProbeAnswerMetric(BaseMetric):

    def __init__(self, conditional: bool = True) -> None:
        self.conditional = conditional
        self.prediction_store = defaultdict(LabelPredictionInstance)
        self.score_store = defaultdict(dict)

    def compute_question_scores(self, group: LabelPredictionInstance) -> Dict[str, float]:

        assert group.label_answer is not None
        assert all([answer is not None for answer in group.probe_predicted_answers])
        assert all([confidence is not None for confidence in group.probe_predicted_confidences])

        probe_predicted_answer = (group.probe_predicted_answers[0]
                                  if group.probe_predicted_confidences[0] > group.probe_predicted_confidences[1]
                                  else group.probe_predicted_answers[1])

        probe_ans_f1, \
        probe_ans_prec, \
        probe_ans_recall = hotpotqa_eval.ans_f1(probe_predicted_answer, group.label_answer)
        probe_ans_em = hotpotqa_eval.ans_em(probe_predicted_answer, group.label_answer)

        if self.conditional:
            assert group.original_predicted_answer is not None, \
            "For conditional evaluation, please pass the prediction from original instance."

            original_ans_f1, \
            original_ans_prec, \
            original_ans_recall = hotpotqa_eval.ans_f1(group.original_predicted_answer,
                                                       group.label_answer)
            original_ans_em = hotpotqa_eval.ans_em(group.original_predicted_answer,
                                                   group.label_answer)

            probe_ans_f1 = min(probe_ans_f1, original_ans_f1)
            probe_ans_prec = min(probe_ans_prec, original_ans_prec)
            probe_ans_recall = min(probe_ans_recall, original_ans_recall)
            probe_ans_em = min(probe_ans_em, original_ans_em)

        question_scores = {"f1": probe_ans_f1, "em": probe_ans_em,
                           "precision": probe_ans_prec, "recall": probe_ans_recall}
        return question_scores

    def store_prediction(self,
                         predicted_answer: str,
                         question_id: str,
                         is_probe: bool,
                         predicted_confidence: float = None,
                         label_answer: str = None) -> None:
        if is_probe:
            assert label_answer is not None
            assert predicted_confidence is not None
            self.prediction_store[question_id].probe_predicted_answers.append(predicted_answer)
            self.prediction_store[question_id].probe_predicted_confidences.append(predicted_confidence)
            self.prediction_store[question_id].label_answer = label_answer
        else:
            self.prediction_store[question_id].original_predicted_answer = predicted_answer

    def reset(self):
        self.prediction_store = defaultdict(LabelPredictionInstance)
        self.score_store = defaultdict(dict)
