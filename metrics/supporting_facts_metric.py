from typing import List, Dict
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy

from metrics import hotpotqa_eval, BaseMetric


@dataclass
class LabelPredictionInstance:
    label_supporting_facts: List = field(default_factory=lambda: deepcopy([]))
    predicted_supporting_facts: List = field(default_factory=lambda: deepcopy([]))


class SupportingFactsMetric(BaseMetric):

    def __init__(self) -> None:
        self.prediction_store = defaultdict(LabelPredictionInstance)
        self.score_store = defaultdict(dict)

    def compute_question_scores(self, group: LabelPredictionInstance) -> Dict[str, float]:

        sp_f1, \
        sp_prec, \
        sp_recall = hotpotqa_eval.sp_f1(group.predicted_supporting_facts, group.label_supporting_facts)
        sp_em = hotpotqa_eval.sp_em(group.predicted_supporting_facts, group.label_supporting_facts)

        question_scores = {"f1": sp_f1, "em": sp_em,
                           "precision": sp_prec, "recall": sp_recall}
        return question_scores

    def store_prediction(self,
                         predicted_supporting_facts: List,
                         label_supporting_facts: List,
                         question_id: str) -> None:
        self.prediction_store[question_id].label_supporting_facts = label_supporting_facts
        self.prediction_store[question_id].predicted_supporting_facts = predicted_supporting_facts

    def reset(self):
        self.prediction_store = defaultdict(LabelPredictionInstance)
        self.score_store = defaultdict(dict)
