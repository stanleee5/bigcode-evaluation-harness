import functools
from concurrent.futures import ThreadPoolExecutor
from typing import List

from datasets import Dataset
from huggingface_hub import InferenceClient
from tqdm import tqdm

from bigcode_eval.base import Task


def tgi_inference(args, prompts: List[str]) -> List[str]:
    client = InferenceClient(model=args.endpoint)

    generate_fn = functools.partial(
        client.text_generation,
        max_new_tokens=args.max_length_generation,
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else None,
        top_p=args.top_p if args.do_sample else None,
        return_full_text=True,
    )

    with ThreadPoolExecutor(32) as executor:
        generations = list(tqdm(executor.map(generate_fn, prompts), total=len(prompts)))

    return generations


def endpoint_inference(
    args,
    task: Task,
    dataset: Dataset,
    n_tasks: int,
) -> List[List[str]]:
    print(f"Endpoint Inference from: {args.endpoint}")

    prompts = []
    start_idx = args.limit_start
    end_idx = start_idx + n_tasks
    for idx in range(start_idx, end_idx):
        prompt_contents = task.get_prompt(dataset[idx])
        if isinstance(prompt_contents, str):
            # Normal code completion mode
            prompt = args.prefix + prompt_contents
        else:
            raise NotImplementedError(
                f"Unsupported prompt format: {type(prompt_contents)}"
            )
        prompts.append(prompt)
    prompts_n_copy = [x for x in prompts for _ in range(args.n_samples)]

    if args.endpoint_type == "tgi":
        generations = tgi_inference(args, prompts_n_copy)
    else:
        raise NotImplementedError(f"check endpoint_type: {args.endpoint_type}")

    generations_list: List[List[str]] = [
        generations[i * args.n_samples : i * args.n_samples + args.n_samples]
        for i in range(len(prompts))
    ]

    code_gens = []
    for _idx, generations in enumerate(generations_list):
        idx = start_idx + _idx  # index from args.limit_start
        code_gens.append(
            [task.postprocess_generation(generation, idx) for generation in generations]
        )

    return code_gens
