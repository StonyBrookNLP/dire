from typing import Dict


class BaseMetric:

    def __init__(self) -> None:
        self.reset()

    def compute_dataset_scores(self):
        total_scores = {"f1": 0.0, "em": 0.0, "precision": 0.0, "recall": 0.0}
        for question_id, question_group in self.prediction_store.items():
            question_scores = self.compute_question_scores(question_group)
            self.score_store[question_id] = question_scores
            for key, value in question_scores.items():
                total_scores[key] += value
        dataset_scores = {name: round(100 * total_score / len(self.prediction_store), 1)
                          if len(self.prediction_store) > 0 else 0.0
                          for name, total_score in total_scores.items()}
        return dataset_scores

    def store_prediction(self, *args):
        raise NotImplementedError

    def compute_question_scores(self, group) -> Dict[str, float]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    @classmethod
    def compute_joint_dataset_scores(cls, metric1, metric2):

        question_ids = metric1.score_store.keys()
        assert set(question_ids) == set(metric2.score_store.keys()), \
            "The two metrics passed in joint_metric call don't have the same question-ids."

        total_scores = {"f1": 0.0, "em": 0.0}
        for question_id in question_ids:

            precision1 = metric1.score_store[question_id]["precision"]
            precision2 = metric2.score_store[question_id]["precision"]
            precision = precision1*precision2
            
            recall1 = metric1.score_store[question_id]["recall"]
            recall2 = metric2.score_store[question_id]["recall"]
            recall = recall1*recall2

            joint_f1 = ((2*precision*recall)/(precision + recall)
                        if precision + recall > 0 else 0.0)
            total_scores["f1"] += joint_f1

            joint_em = min(metric1.score_store[question_id]["em"],
                              metric2.score_store[question_id]["em"])
            total_scores["em"] += joint_em

        dataset_scores = {name: round(100 * total_score / len(question_ids), 1)
                          if len(question_ids) > 0 else 0.0
                          for name, total_score in total_scores.items()}
        return dataset_scores
