from pprint import pprint
from typing import List, Union
import inspect

import sacrebleu
import lm_eval.base

from . import superglue
from . import glue
from . import arc
from . import coqa
from . import race
from . import webqs
from . import anli
from . import wsc273
from . import winogrande
from . import quac
from . import hellaswag
from . import swag
from . import openbookqa
from . import squad
from . import naturalqs
from . import sat
from . import arithmetic
from . import lambada
from . import piqa
from . import prost
from . import mc_taco
from . import triviaqa
from . import pubmedqa
from . import sciq
from . import qasper
from . import qa4mre
from . import translation
from . import headqa
from . import mathqa
from . import hendrycks_ethics
from . import drop
from . import unscramble
from . import logiqa
from . import hendrycks_test
from . import hendrycks_math
from . import cbt
from . import lambada_cloze
from . import pile
from . import wikitext
from . import lambada_multilingual
from . import mutual
from . import truthfulqa
from . import blimp
from . import asdiv
from . import gsm8k
from . import storycloze
from . import toxigen
from . import crowspairs
from .ja import jsquad
from .ja import jaquad
from .ja import jcommonsenseqa
from .ja import jnli
from .ja import marc_ja
from .ja import jcola
from .ja import jblimp
from .ja import wikilingua
from .ja import xwinograd
from .ja import xlsum_ja
from .ja import jaqket_v1
from .ja import jaqket_v2
from .ja import mgsm

########################################
# Translation tasks
########################################

# 6 total
gpt3_translation_benchmarks = {
    "wmt14": ["en-fr", "fr-en"],  # French
    "wmt16": ["en-ro", "ro-en", "de-en", "en-de"],  # German, Romanian
}


# 28 total
selected_translation_benchmarks = {
    **gpt3_translation_benchmarks,
    "wmt20": sacrebleu.get_langpairs_for_testset("wmt20"),
    "iwslt17": ["en-ar", "ar-en"],  # Arabic
}

# 319 total
all_translation_benchmarks = {
    ts: sacrebleu.get_langpairs_for_testset(ts)
    for ts in sacrebleu.get_available_testsets()
}


########################################
# All tasks
########################################


TASK_REGISTRY = {}


# register all the subclasses of Task
def register_tasks():
    qq = []
    qq.extend(lm_eval.base.Task.__subclasses__())
    while qq:
        cls = qq.pop()
        # add subclasses to recur
        qq.extend(cls.__subclasses__())

        # get the shortname using the module
        mod = inspect.getmodule(cls)
        name = mod.__name__.split(".")[-1]
        # only the first one gets added as a plain name
        if name not in TASK_REGISTRY:
            TASK_REGISTRY[name] = cls

        if hasattr(cls, "PROMPT_VERSION"):
            # Note that anything with a prompt version has a VERSION
            key = f"{name}-{cls.VERSION}-{cls.PROMPT_VERSION}"
            TASK_REGISTRY[key] = cls


register_tasks()


ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def get_task_dict(task_name_list: List[Union[str, lm_eval.base.Task]]):
    task_name_dict = {
        task_name: get_task(task_name)()
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if not isinstance(task_object, str)
    }
    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {**task_name_dict, **task_name_from_object_dict}
