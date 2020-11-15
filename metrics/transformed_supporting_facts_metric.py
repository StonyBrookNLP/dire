from typing import List, Dict
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy

from metrics import hotpotqa_eval, BaseMetric


@dataclass
class LabelPredictionInstance:
    label_supporting_facts: List = field(default_factory=lambda: deepcopy([]))
    predicted_supporting_facts: List = field(default_factory=lambda: deepcopy([]))

    label_sufficiencies: List = field(default_factory=lambda: deepcopy([]))
    predicted_sufficiencies: List = field(default_factory=lambda: deepcopy([]))


class TransformedSupportingFactsMetric(BaseMetric):

    def __init__(self, with_sufficiency: bool = True) -> None:
        self.prediction_store = defaultdict(LabelPredictionInstance)
        self.score_store = defaultdict(dict)
        self.with_sufficiency = with_sufficiency

    def compute_question_scores(self, group: LabelPredictionInstance) -> Dict[str, float]:

        sp_f1, sp_prec, sp_recall = hotpotqa_eval.sp_f1(group.predicted_supporting_facts,
                                                        group.label_supporting_facts)
        sp_em = hotpotqa_eval.sp_em(group.predicted_supporting_facts, group.label_supporting_facts)

        if self.with_sufficiency:
            assert len(group.predicted_sufficiencies) == 3
            sufficiency_score = group.predicted_sufficiencies == group.label_sufficiencies
            sp_f1 = sp_f1 if sufficiency_score else 0.0
            sp_prec = sp_prec if sufficiency_score else 0.0
            sp_recall = sp_recall if sufficiency_score else 0.0
            sp_em = sp_em if sufficiency_score else 0.0

        question_scores = {"f1": sp_f1, "em": sp_em,
                           "precision": sp_prec, "recall": sp_recall}
        return question_scores

    def store_prediction(self,
                         predicted_supporting_facts: List,
                         label_supporting_facts: List,
                         label_sufficiency: int,
                         question_id: str,
                         predicted_sufficiency: int = None) -> None:
        if label_sufficiency == 1:
            self.prediction_store[question_id].label_supporting_facts = label_supporting_facts
            self.prediction_store[question_id].predicted_supporting_facts = predicted_supporting_facts

        if self.with_sufficiency:
            assert predicted_sufficiency is not None
            self.prediction_store[question_id].predicted_sufficiencies.append(predicted_sufficiency)
            self.prediction_store[question_id].label_sufficiencies.append(label_sufficiency)

    def reset(self):
        self.prediction_store = defaultdict(LabelPredictionInstance)
        self.score_store = defaultdict(dict)
