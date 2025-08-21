# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/gsm8k")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "openai/gsm8k"

    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = """### **INTERNAL REASONING**

WRITE STEP-BY-STEP 

FINALLY PROVIDE USER ANSWER BETWEEN <answer>...</answer>

**Step 1: Parse the Question**

* What is the question asking?
* What format or depth of answer is expected?
* What is the user’s supplied answer (if any)?

**Step 2: Retrieve Relevant Knowledge**

* Write your relevant prior knowledge, facts, learnings, formulae, theories, memories, or context related to the question.

* Write you used that relevant knowledge. 

**Step 3: Generate Multiple Candidate Answer Pathways**

* Outline at least two plausible reasoning paths that could answer the question.
* Briefly summarize each pathway.

**Step 4: Contradiction Audit**

* Compare the user’s supplied answer with prior knowledge.
* Identify any conflicts or contradictions.
* Analyze root causes of contradictions (e.g., differences in definitions, outdated info, missing premises).

**Step 5: Trace Logical Steps**

* For the chosen pathway, explicitly link facts, inferences, and sub-conclusions.
* Show why each step logically follows from the question or evidence.

**Step 6: Explore Alternative Explanations**

* Note other plausible interpretations or answers and why they might be considered.

**Step 7: Decision and Motivation**

* Select the best answer or course of action.
* Explain why this choice beats alternatives, referencing evidence, relevance, clarity, and value considerations (e.g., politeness, safety).
* If adopting the user’s answer, specify what insight led to this change.
* If uncertain, justify why a clarifying question is needed instead of guessing.

**Step 8: Confidence and Error Monitoring**

* Rate confidence as High, Medium, or Low.
* Note one plausible weakness or assumption that might make the answer incorrect.

**Step 9: Reflect on Learning**

* Briefly reflect on what new knowledge, patterns, or reasoning strategies were gained from this example.
* Suggest how future similar questions might be answered better.

---

### **FINAL USER REPLY (DELIVER THIS TO THE USER)**

* Provide a concise, clear, and helpful answer or clarifying question in plain language.
* Avoid jargon, internal tags, or technical explanations.

---"""

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = instruction_following + " " + question_raw

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
