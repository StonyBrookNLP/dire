from copy import deepcopy
from typing import List, Dict
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy

from metrics import hotpotqa_eval, BaseMetric


@dataclass
class LabelPredictionInstance:
    label_supporting_facts: List = field(default_factory=lambda: deepcopy([]))
    probe_predicted_supporting_facts: List = field(default_factory=lambda: deepcopy([]))

    label_sufficiencies: List = field(default_factory=lambda: deepcopy([]))
    predicted_sufficiencies: List = field(default_factory=lambda: deepcopy([]))


class ProbeTransformedSupportingFactsMetric(BaseMetric):

    def __init__(self, with_sufficiency: bool = True) -> None:
        self.prediction_store = defaultdict(LabelPredictionInstance)
        self.score_store = defaultdict(dict)
        self.with_sufficiency = with_sufficiency

    def compute_question_scores(self, group: LabelPredictionInstance) -> Dict[str, float]:

        assert len(group.probe_predicted_supporting_facts) == 2

        probe_predicted_supporting_facts = group.probe_predicted_supporting_facts[0] + \
                                           group.probe_predicted_supporting_facts[1]

        probe_sp_f1, \
        probe_sp_prec, \
        probe_sp_recall = hotpotqa_eval.sp_f1(probe_predicted_supporting_facts, group.label_supporting_facts)
        probe_sp_em = hotpotqa_eval.sp_em(probe_predicted_supporting_facts, group.label_supporting_facts)

        if self.with_sufficiency:
            assert len(group.predicted_sufficiencies) == 3
            sufficiency_score = group.predicted_sufficiencies == group.label_sufficiencies
            probe_sp_f1 = probe_sp_f1 if sufficiency_score else 0.0
            probe_sp_prec = probe_sp_prec if sufficiency_score else 0.0
            probe_sp_recall = probe_sp_recall if sufficiency_score else 0.0
            probe_sp_em = probe_sp_em if sufficiency_score else 0.0

        question_scores = {"f1": probe_sp_f1, "em": probe_sp_em,
                           "precision": probe_sp_prec, "recall": probe_sp_recall}
        return question_scores

    def store_prediction(self,
                         predicted_supporting_facts: List,
                         label_supporting_facts: List,
                         question_id: str,
                         label_sufficiency: int,
                         predicted_sufficiency: int = None) -> None:

        if label_sufficiency == 0:
            self.prediction_store[question_id].probe_predicted_supporting_facts.append(predicted_supporting_facts)
            self.prediction_store[question_id].label_supporting_facts.extend(label_supporting_facts)

        if self.with_sufficiency:
            assert predicted_sufficiency is not None
            self.prediction_store[question_id].predicted_sufficiencies.append(predicted_sufficiency)
            self.prediction_store[question_id].label_sufficiencies.append(label_sufficiency)

    def reset(self):
        self.prediction_store = defaultdict(LabelPredictionInstance)
        self.score_store = defaultdict(dict)
