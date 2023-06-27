import os
import argparse
import json
import logging
import fnmatch

from lm_eval import tasks, evaluator

logging.getLogger("openai").setLevel(logging.WARNING)


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=str, default="0")
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=str, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


# Returns a dict containing the keys and values of the source_dict that
# the key of which matchs at least one of the patterns
def pattern_match(patterns, source_dict):
    patterns = sort(patterns) # sorted by task name
    source_list = list(source_dict.keys())
    task_param_dict = dict()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_param_dict.update({matching: source_dict[matching]})
    return task_param_dict


def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        _tasks = tasks.ALL_TASKS
    else:
        _tasks = args.tasks.split(",")

    if "," in args.num_fewshot:
        _num_fewshot = [int(n) for n in args.num_fewshot.split(",")]
    else:
        _num_fewshot = [int(args.num_fewshot) for _ in _tasks]
    
    if args.limit is not None:
        if "," in args.limit:
            _limit = [int(n) if n.isdigit() else float(n) for n in args.limit.split(",")]
        else:
            _limit = [int(args.limit) for _ in _tasks]
    else:
        _limit = [None for _ in _tasks]

    task_param_dict = {tup[0]: tup for tup in zip(_tasks, _num_fewshot, _limit)}
    task_param_dict = pattern_match(tasks.ALL_TASKS, task_param_dict)

    task_names = [tup[0] for tup in task_param_dict.values()]
    num_fewshot = [tup[1] for tup in task_param_dict.values()]
    limit = [tup[2] for tup in task_param_dict.values()]
    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)
    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        verbose=args.verbose,
    )

    dumped = json.dumps(results, indent=2, ensure_ascii=False)
    print(dumped)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    )
    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
