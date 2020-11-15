from typing import List, Dict
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy

from metrics import hotpotqa_eval, BaseMetric


@dataclass
class LabelPredictionInstance:
    label_answer: str = None
    predicted_answer: str = None

    label_sufficiencies: List = field(default_factory=lambda: deepcopy([]))
    predicted_sufficiencies: List = field(default_factory=lambda: deepcopy([]))


class TransformedAnswerMetric(BaseMetric):

    def __init__(self, with_sufficiency: bool = True) -> None:
        self.prediction_store = defaultdict(LabelPredictionInstance)
        self.score_store = defaultdict(dict)
        self.with_sufficiency = with_sufficiency

    def compute_question_scores(self, group: LabelPredictionInstance) -> Dict[str, float]:

        assert group.label_answer is not None
        assert group.predicted_answer is not None

        ans_f1, ans_prec, ans_recall = hotpotqa_eval.ans_f1(group.predicted_answer,
                                                            group.label_answer)
        ans_em = hotpotqa_eval.ans_em(group.predicted_answer, group.label_answer)

        if self.with_sufficiency:
            assert len(group.predicted_sufficiencies) == 3
            sufficiency_score = group.predicted_sufficiencies == group.label_sufficiencies
            ans_f1 = ans_f1 if sufficiency_score else 0.0
            ans_prec = ans_prec if sufficiency_score else 0.0
            ans_recall = ans_recall if sufficiency_score else 0.0
            ans_em = ans_em if sufficiency_score else 0.0

        question_scores = {"f1": ans_f1, "em": ans_em,
                           "precision": ans_prec, "recall": ans_recall}
        return question_scores

    def store_prediction(self,
                         predicted_answer: str,
                         label_answer: str,
                         label_sufficiency: int,
                         question_id: str,
                         predicted_sufficiency: int = None) -> None:
        if label_sufficiency == 1:
            self.prediction_store[question_id].predicted_answer = predicted_answer
            self.prediction_store[question_id].label_answer = label_answer

        if self.with_sufficiency:
            assert predicted_sufficiency is not None
            self.prediction_store[question_id].predicted_sufficiencies.append(predicted_sufficiency)
            self.prediction_store[question_id].label_sufficiencies.append(label_sufficiency)

    def reset(self):
        self.prediction_store = defaultdict(LabelPredictionInstance)
        self.score_store = defaultdict(dict)
