from copy import deepcopy
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

    label_sufficiencies: List = field(default_factory=lambda: deepcopy([]))
    predicted_sufficiencies: List = field(default_factory=lambda: deepcopy([]))


class ProbeTransformedAnswerMetric(BaseMetric):

    def __init__(self, with_sufficiency: bool = True) -> None:
        self.prediction_store = defaultdict(LabelPredictionInstance)
        self.score_store = defaultdict(dict)
        self.with_sufficiency = with_sufficiency

    def compute_question_scores(self, group: LabelPredictionInstance) -> Dict[str, float]:

        assert group.label_answer is not None
        assert all([answer is not None for answer in group.probe_predicted_answers])
        assert all([confidence is not None for confidence in group.probe_predicted_confidences])
        assert len(group.probe_predicted_answers) == len(group.probe_predicted_confidences) == 2

        probe_predicted_answer = (group.probe_predicted_answers[0]
                                  if group.probe_predicted_confidences[0] > group.probe_predicted_confidences[1]
                                  else group.probe_predicted_answers[1])

        probe_ans_f1, \
        probe_ans_prec, \
        probe_ans_recall = hotpotqa_eval.ans_f1(probe_predicted_answer, group.label_answer)
        probe_ans_em = hotpotqa_eval.ans_em(probe_predicted_answer, group.label_answer)

        if self.with_sufficiency:
            assert len(group.predicted_sufficiencies) == 3
            sufficiency_score = group.predicted_sufficiencies == group.label_sufficiencies
            probe_ans_f1 = probe_ans_f1 if sufficiency_score else 0.0
            probe_ans_prec = probe_ans_prec if sufficiency_score else 0.0
            probe_ans_recall = probe_ans_recall if sufficiency_score else 0.0
            probe_ans_em = probe_ans_em if sufficiency_score else 0.0

        question_scores = {"f1": probe_ans_f1, "em": probe_ans_em,
                           "precision": probe_ans_prec, "recall": probe_ans_recall}
        return question_scores

    def store_prediction(self,
                         predicted_answer: str,
                         predicted_confidence: float,
                         label_answer: str,
                         label_sufficiency: int,
                         question_id: str,
                         predicted_sufficiency: int = None) -> None:

        self.prediction_store[question_id].label_answer = label_answer

        if label_sufficiency == 0:
            self.prediction_store[question_id].probe_predicted_answers.append(predicted_answer)
            self.prediction_store[question_id].probe_predicted_confidences.append(predicted_confidence)

        if self.with_sufficiency:
            assert predicted_sufficiency is not None
            self.prediction_store[question_id].predicted_sufficiencies.append(predicted_sufficiency)
            self.prediction_store[question_id].label_sufficiencies.append(label_sufficiency)

    def reset(self):
        self.prediction_store = defaultdict(LabelPredictionInstance)
        self.score_store = defaultdict(dict)
