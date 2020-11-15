from typing import List, Dict
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy

from metrics import hotpotqa_eval, BaseMetric


@dataclass
class LabelPredictionInstance:
    label_supporting_facts: List = field(default_factory=lambda: deepcopy([]))

    probe_predicted_supporting_facts: List = field(default_factory=lambda: deepcopy([]))

    original_predicted_supporting_facts: List = field(default_factory=lambda: deepcopy([]))


class ProbeSupportingFactsMetric(BaseMetric):

    def __init__(self, conditional: bool = True) -> None:
        self.conditional = conditional
        self.prediction_store = defaultdict(LabelPredictionInstance)
        self.score_store = defaultdict(dict)

    def compute_question_scores(self, group: LabelPredictionInstance) -> Dict[str, float]:

        assert len(group.probe_predicted_supporting_facts) == 2

        probe_predicted_supporting_facts = group.probe_predicted_supporting_facts[0] + \
                                           group.probe_predicted_supporting_facts[1]

        probe_sp_f1, \
        probe_sp_prec, \
        probe_sp_recall = hotpotqa_eval.sp_f1(probe_predicted_supporting_facts, group.label_supporting_facts)
        probe_sp_em = hotpotqa_eval.sp_em(probe_predicted_supporting_facts, group.label_supporting_facts)

        if self.conditional:
            assert group.original_predicted_supporting_facts is not None, \
            "For conditional evaluation, please pass the prediction from original instance."

            original_sp_f1, \
            original_sp_prec, \
            original_sp_recall = hotpotqa_eval.sp_f1(group.original_predicted_supporting_facts,
                                                     group.label_supporting_facts)
            original_sp_em = hotpotqa_eval.sp_em(group.original_predicted_supporting_facts,
                                                 group.label_supporting_facts)

            probe_sp_f1 = min(probe_sp_f1, original_sp_f1)
            probe_sp_prec = min(probe_sp_prec, original_sp_prec)
            probe_sp_recall = min(probe_sp_recall, original_sp_recall)
            probe_sp_em = min(probe_sp_em, original_sp_em)

        question_scores = {"f1": probe_sp_f1, "em": probe_sp_em,
                           "precision": probe_sp_prec, "recall": probe_sp_recall}
        return question_scores

    def store_prediction(self,
                         predicted_supporting_facts: List,
                         question_id: str,
                         is_probe: bool,
                         label_supporting_facts: List = None) -> None:
        if is_probe:
            assert label_supporting_facts is not None
            self.prediction_store[question_id].probe_predicted_supporting_facts.append(predicted_supporting_facts)
            self.prediction_store[question_id].label_supporting_facts.extend(label_supporting_facts)
        else:
            self.prediction_store[question_id].original_predicted_supporting_facts = predicted_supporting_facts

    def reset(self):
        self.prediction_store = defaultdict(LabelPredictionInstance)
        self.score_store = defaultdict(dict)
