#!/usr/bin/env python
# Run a suite of tests

import argparse
import configparser
from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path

from lm_eval import tasks, evaluator

# get path of current file
FILE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
# Path to suite configs
SUITE_DIR = FILE_PATH / "../lm_eval/suites"

# names for prompts
# TODO move this into lm_eval
PROMPT_CODES = {
    "user": "0.0",
    "jgpt": "0.1",
    "fintan": "0.2",
    "fintan2": "0.2.1",
    "ja-alpaca": "0.3",
    "rinna-sft": "0.4",
    "rinna-bilingual": "0.5",
    "llama2": "0.6",
}


@dataclass
class TaskSpec:
    """Specification of a task in an eval suite.

    A suite is a list of these specs, plus a prompt."""

    # The real arguments have to be massaged into messy strings and parallel
    # lists, but this is a more reasonable structure - we can handle conversion
    # separately.

    name: str
    fewshot: int
    version: Optional[str]


def build_eval_args(specs: list[TaskSpec], prompt: str) -> tuple[list[str], list[int]]:
    """Convert list of TaskSpecs into args for simple_evaluate."""

    tasks = []
    fewshot = []
    prompt_code = PROMPT_CODES[prompt]
    for spec in specs:
        task_name = spec.name
        if spec.version is not None:
            task_name += "-" + spec.version + "-" + prompt_code

        tasks.append(task_name)
        fewshot.append(spec.fewshot)

    return (tasks, fewshot)


def load_suite(name):
    """Read in configuration for a test suite.

    A suite will have a config file named something like `my_suite.conf`. For
    each task in the file, a version, fewshot config, and any other details
    will be specified.

    Example entry:

        [tasks.mgsm]
        version = 1.0
        fewshot = 5
    """
    conf = configparser.ConfigParser()
    conf.read(SUITE_DIR / (name + ".conf"))

    specs = []
    for key, val in conf.items():
        if not key.startswith("tasks."):
            continue

        spec = TaskSpec(
            name=key.split(".", 1)[1],
            version=val.get("version", None),
            fewshot=int(val["fewshot"]),
        )
        specs.append(spec)
    return specs


def run_suite(
    model_args, suite, prompt, *, model_type="hf-causal", output=None, verbose=False
):
    # Confusing detail: in the "simple evaluate", "model" is the HF model type,
    # which is almost always hf-causal or hf-causal-experimental. `model_args`
    # looks like this:
    #
    #     pretrained=hoge/piyo,tokenizer=...,asdf=...

    # device never changes in practice
    device = "cuda"

    print("suite", suite)
    specs = load_suite(suite)
    print(specs)
    tasks, num_fewshot = build_eval_args(specs, prompt)
    print(tasks)
    print(num_fewshot)
    print(model_args)

    evaluator.simple_evaluate(
        model=model_type,
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        device=device,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="run_suite.py", description="Run a test suite with a model"
    )
    parser.add_argument("model", help="Model path (or HF spec)")
    parser.add_argument("suite", help="Test suite to run")
    parser.add_argument("prompt", help="Prompt to use")
    parser.add_argument("-m", "--model_args", help="Additional model arguments")
    parser.add_argument(
        "-t", "--model_type", default="hf-causal-experimental", help="Model type"
    )
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    margs = f"pretrained={args.model}"
    if args.model_args:
        margs = args.model + "," + args.model_args

    run_suite(
        margs,
        args.suite,
        args.prompt,
        model_type=args.model_type,
        output=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
