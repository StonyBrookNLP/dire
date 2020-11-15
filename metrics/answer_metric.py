from typing import List, Dict
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy


from metrics import hotpotqa_eval, BaseMetric


@dataclass
class LabelPredictionInstance:
    label_answer: str = None
    predicted_answer: str = None


class AnswerMetric(BaseMetric):

    def __init__(self) -> None:
        self.prediction_store = defaultdict(LabelPredictionInstance)
        self.score_store = defaultdict(dict)

    def compute_question_scores(self, group: LabelPredictionInstance) -> Dict[str, float]:

        assert group.label_answer is not None
        assert group.predicted_answer is not None

        ans_f1, ans_prec, ans_recall = hotpotqa_eval.ans_f1(group.predicted_answer,
                                                            group.label_answer)
        ans_em = hotpotqa_eval.ans_em(group.predicted_answer, group.label_answer)
        question_scores = {"f1": ans_f1, "em": ans_em,
                           "precision": ans_prec, "recall": ans_recall}
        return question_scores

    def store_prediction(self,
                         predicted_answer: str,
                         label_answer: str,
                         question_id: str) -> None:
        self.prediction_store[question_id].label_answer = label_answer
        self.prediction_store[question_id].predicted_answer = predicted_answer

    def reset(self):
        self.prediction_store = defaultdict(LabelPredictionInstance)
        self.score_store = defaultdict(dict)
