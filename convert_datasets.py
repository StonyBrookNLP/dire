"""
Takes Raw HotpotQA dataset files and generates:
    1. DiRe Probe of HotpotQA
    2. CSS Transformed dataset of HotpotQA
    3. DiRe Probe of the CSS Transformed dataset of HotpotQA
"""

import json
import copy
from typing import List, Dict
import random

from tqdm import tqdm
import numpy as np


# DO NOT CHANGE THE SEED.
random.seed(13370)
np.random.seed(13370)
# Random calls to generate sub_idx were introduced after paper submission.
# We've used np.random instead of random only to not interfere
# with the random-states of the original processing script.


def delete_paragraph(instance: Dict, paragraph_title: str) -> None:
    index = [paragraph[0] for paragraph in instance["context"]].index(paragraph_title)
    instance["context"].pop(index)

def replace_paragraph(instance: Dict, original_paragraph_title: str, updated_paragraph: List) -> None:
    index = [paragraph[0] for paragraph in instance["context"]].index(original_paragraph_title)
    instance["context"][index] = updated_paragraph

def delete_supporting_paragraph(instance: Dict,
                                supporting_paragraph_titles: List[str]) -> None:
    instance["supporting_facts"] = [info for info in instance["supporting_facts"]
                                    if info[0] in supporting_paragraph_titles]

def write_instances_to_file_path(instances, file_path):
    with open(file_path, "w") as file:
        file.write(json.dumps(instances, indent=4))

def generate_probe_of_original_instance(instance: Dict):

    supporting_facts_info = original_instance["supporting_facts"]                
    supporting_paragraph_titles = list({info[0] for info in supporting_facts_info})

    instance1 = copy.deepcopy(original_instance)
    instance2 = copy.deepcopy(original_instance)

    delete_paragraph(instance1, supporting_paragraph_titles[0])
    delete_supporting_paragraph(instance1, supporting_paragraph_titles[0])

    delete_paragraph(instance2, supporting_paragraph_titles[1])
    delete_supporting_paragraph(instance2, supporting_paragraph_titles[1])

    return [instance1, instance2]

def generate_transformed_instance(instance: Dict,
                                  replacement_pargraph: List,
                                  balance: bool = False):

    supporting_facts_info = original_instance["supporting_facts"]                
    supporting_paragraph_titles = list({info[0] for info in supporting_facts_info})
    replacement_pargraph_title = replacement_pargraph[0]

    instance1 = copy.deepcopy(original_instance)
    instance2 = copy.deepcopy(original_instance)
    instance3 = copy.deepcopy(original_instance)
    start_sub_idx = int(np.random.choice(3, 1)[0])

    delete_paragraph(instance1, replacement_pargraph_title)
    instance1["sufficiency"] = 1
    instance1["sub_idx"] = (start_sub_idx)%3

    delete_paragraph(instance2, replacement_pargraph_title)
    replace_paragraph(instance2, supporting_paragraph_titles[0], replacement_pargraph)
    delete_supporting_paragraph(instance2, supporting_paragraph_titles[1])
    instance2["sufficiency"] = 0
    instance2["sub_idx"] = (start_sub_idx+1)%3

    delete_paragraph(instance3, replacement_pargraph_title)
    replace_paragraph(instance3, supporting_paragraph_titles[1], replacement_pargraph)
    delete_supporting_paragraph(instance3, supporting_paragraph_titles[0])
    instance3["sufficiency"] = 0
    instance3["sub_idx"] = (start_sub_idx+2)%3

    assert len(instance1["context"]) == len(instance2["context"]) == len(instance3["context"])

    transformed_instances = [instance1]

    if balance:
        if random.random() > 0.5:
            transformed_instances.append(instance2)
        else:
            transformed_instances.append(instance3)
    else:
        transformed_instances.append(instance2)
        transformed_instances.append(instance3)

    return transformed_instances

def generate_probe_of_transformed_instance(instance: Dict,
                                           replacement_pargraph: List,
                                           balance: bool = False):

    supporting_facts_info = original_instance["supporting_facts"]                
    supporting_paragraph_titles = list({info[0] for info in supporting_facts_info})
    replacement_pargraph_title = replacement_pargraph[0]

    instance1 = copy.deepcopy(original_instance)
    instance2 = copy.deepcopy(original_instance)
    instance3 = copy.deepcopy(original_instance)
    start_sub_idx = int(np.random.choice(3, 1)[0])

    delete_paragraph(instance1, replacement_pargraph_title)
    delete_paragraph(instance1, supporting_paragraph_titles[0])
    delete_supporting_paragraph(instance1, supporting_paragraph_titles[0])
    instance1["sufficiency"] = 0
    instance1["sub_idx"] = (start_sub_idx)%3

    delete_paragraph(instance2, replacement_pargraph_title)
    delete_paragraph(instance2, supporting_paragraph_titles[1])
    delete_supporting_paragraph(instance2, supporting_paragraph_titles[1])
    instance2["sufficiency"] = 0
    instance2["sub_idx"] = (start_sub_idx+1)%3

    delete_paragraph(instance3, replacement_pargraph_title)
    replace_paragraph(instance3, supporting_paragraph_titles[0], replacement_pargraph)
    delete_paragraph(instance3, supporting_paragraph_titles[1])
    delete_supporting_paragraph(instance2, supporting_paragraph_titles)
    instance3["sufficiency"] = -1
    instance3["sub_idx"] = (start_sub_idx+2)%3

    probe_of_transformed_instances = [instance3]
    if balance:
        # To balance training dataset according sufficiency label.
        if random.random() > 0.5:
            probe_of_transformed_instances.append(instance1)
        else:
            probe_of_transformed_instances.append(instance2)
    else:
        probe_of_transformed_instances.append(instance1)
        probe_of_transformed_instances.append(instance2)
    return probe_of_transformed_instances


if __name__ == '__main__':

    raw_hotpotqa_file_paths = ["data/raw/hotpot_train_v1.1.json",
                               "data/raw/hotpot_dev_distractor_v1.json",
                               "data/raw/example.json"]


    original_dataset_file_paths = [f"data/processed/original_hotpotqa_train.json",
                                   f"data/processed/original_hotpotqa_dev.json",
                                   f"tests/fixtures/datasets/hotpotqa_fixture.json"]

    probe_of_original_dataset_file_paths = [f"data/processed/probe_of_original_hotpotqa_train.json",
                                            f"data/processed/probe_of_original_hotpotqa_dev.json",
                                            f"tests/fixtures/datasets/probe_of_hotpotqa_fixture.json"]

    transformed_dataset_file_paths = [f"data/processed/transformed_hotpotqa_train.json",
                                      f"data/processed/transformed_hotpotqa_dev.json",
                                      f"tests/fixtures/datasets/transformed_hotpotqa_fixture.json"]

    probe_of_transformed_dataset_file_paths = [f"data/processed/probe_of_transformed_hotpotqa_train.json",
                                               f"data/processed/probe_of_transformed_hotpotqa_dev.json",
                                               f"tests/fixtures/datasets/probe_of_transformed_hotpotqa_fixture.json"]


    for raw_hotpotqa_file_path, \
        original_dataset_file_path, \
        probe_of_original_dataset_file_path, \
        transformed_dataset_file_path, \
        probe_of_transformed_dataset_file_path in zip(raw_hotpotqa_file_paths,
                                                      original_dataset_file_paths,
                                                      probe_of_original_dataset_file_paths,
                                                      transformed_dataset_file_paths,
                                                      probe_of_transformed_dataset_file_paths):

        skipped_instances = 0

        original_dataset_instances = []
        probe_of_original_dataset_instances = []
        transformed_dataset_instances = []
        probe_of_transformed_dataset_instances = []

        print("\nReading from: " + raw_hotpotqa_file_path)
        with open(raw_hotpotqa_file_path, "r") as read_file:

            instances = json.load(read_file)

            tqdm_object = tqdm(instances)
            for ii, original_instance in enumerate(tqdm_object):
                original_instance = copy.deepcopy(original_instance)
                paragraphs = original_instance["context"]

                # Skip instances with less than 5 total paragraphs.
                if len(paragraphs) < 5:
                    skipped_instances += 1
                    tqdm_object.set_description(f"Skipped: {skipped_instances} instances with < 5 paragraphs.")
                    continue

                # 1. Original Instance: Use unskipped instances
                # (~0.5% are only skipped because of < 5 paras).
                original_dataset_instances.append(original_instance)

                # 2. Probe of Original Instance
                probe_of_original_dataset_instances.extend(
                    generate_probe_of_original_instance(original_instance)
                )

                balance = "train" in original_dataset_file_path or "fixture" in original_dataset_file_path

                paragraph2text = lambda paragraph: " ".join(paragraph[1]).strip()
                supporting_paragraph_titles = dict(original_instance["supporting_facts"]).keys()
                supporting_paragraph_texts = [paragraph2text(paragraph) for paragraph in paragraphs
                                              if paragraph[0] in supporting_paragraph_titles]
                non_supporting_paragraphs = [paragraph for paragraph in paragraphs
                                             if paragraph2text(paragraph) not in supporting_paragraph_texts]
                replacement_pargraph = random.sample(non_supporting_paragraphs, 1)[0]

                # 3. Transformed Instance
                transformed_dataset_instances.extend(
                    generate_transformed_instance(original_instance, replacement_pargraph, balance=balance)
                )

                # 4. Probe of Transformed Instance
                probe_of_transformed_dataset_instances.extend(
                    generate_probe_of_transformed_instance(original_instance, replacement_pargraph, balance=balance)
                )

        print(f"Writing in {original_dataset_file_path}")
        write_instances_to_file_path(original_dataset_instances, original_dataset_file_path)

        print(f"Writing in {probe_of_original_dataset_file_path}")
        write_instances_to_file_path(probe_of_original_dataset_instances, probe_of_original_dataset_file_path)

        print(f"Writing in {transformed_dataset_file_path}")
        write_instances_to_file_path(transformed_dataset_instances, transformed_dataset_file_path)

        print(f"Writing in {probe_of_transformed_dataset_file_path}")
        write_instances_to_file_path(probe_of_transformed_dataset_instances, probe_of_transformed_dataset_file_path)

        skip_percentage = 100*(skipped_instances) / len(instances)
        print(f"Skipped {round(skip_percentage, 1)}% instances from {raw_hotpotqa_file_path}")
