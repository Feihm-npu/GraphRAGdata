# extract.py
from datasets import load_dataset
import argparse, json
from pathlib import Path
from tqdm.auto import tqdm
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompts import PROMPTS
import re
import time
# --- prompt 常量 ---
SYS_PROMPT = (
    "You are a helpful assistant for extracting entities and relationships "
    "for a GraphRAG task."
)
TUPLE_DELIM = "|"
RECORD_DELIM = "<REC>"
COMPLETION_TOKEN = "<ENTITY EXTRACTED>"

# 你需要的实体类型（示例，可自行修改）
ENTITY_TYPES = "person, organization, location, technology, role, event"

# --------------------------------------------------------------------------- #
def build_prompt(example, tokenizer):
    """把 NQ 单条样本转成 chat-prompt 字符串"""
    ctxs = example["ctxs"]
    snippet_tpl = "# Title\n{title}\n\n## Text\n{text}"
    # 拼接 passage 文本
    passages = "\n\n".join(
        snippet_tpl.format(title=c.get("title", "<title>"), text=c.get("text", "<text>"))
        for c in ctxs
    )

    user_prompt = PROMPTS["entity_extraction"].format(
        tuple_delimiter=TUPLE_DELIM,
        record_delimiter=RECORD_DELIM,
        input_text=passages,
        entity_types=ENTITY_TYPES,
        completion_delimiter=COMPLETION_TOKEN,
    )

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def parse_triples(raw_text, tuple_delimiter="|", completion_delimiter="<ENTITY EXTRACTED>"):
    triples = []
    raw_text = raw_text.strip()

    if completion_delimiter in raw_text:
        raw_text = raw_text.split(completion_delimiter)[0]

    # 允许字段加不加引号
    pattern = re.compile(
        r'\(\s*"(entity|relationship)"\s*' +
        re.escape(tuple_delimiter) + r'\s*("?)([^"|]+)\2' +
        re.escape(tuple_delimiter) + r'\s*("?)([^"|]+)\4' +
        re.escape(tuple_delimiter) + r'\s*("?)([^"|]+)\6' +
        r'(?:' + re.escape(tuple_delimiter) + r'\s*("?)([^"|)]+)\8)?\s*\)'
    )

    for match in pattern.finditer(raw_text):
        kind = match.group(1)

        if kind == "entity":
            triples.append({
                "type": "entity",
                "name": match.group(3).strip(),
                "entity_type": match.group(5).strip(),
                "desc": match.group(7).strip(),
            })
        elif kind == "relationship":
            score_str = match.group(9)
            try:
                score = int(score_str.strip())
            except:
                score = 5
            triples.append({
                "type": "relation",
                "head": match.group(3).strip(),
                "tail": match.group(5).strip(),
                "desc": match.group(7).strip(),
                "score": score,
            })

    return triples


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True, help="Path to nq.json")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--world_size", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--dest_dir", required=True)
    ap.add_argument("--num_proc", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    return ap.parse_args()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    args = parse_args()

    print("[+] Loading dataset ...")
    ds = load_dataset("json", data_files=args.data_file, split="train").select(range(9600))
    print(f"    Dataset size = {len(ds)}")

    print("[+] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False)

    # 预生成 prompt 字符串
    print("[+] Building prompts in parallel ...")
    ds = ds.map(
        lambda ex: {
            "prompt": build_prompt(ex, tokenizer),
            "question": ex["question"],
            "ctxs": ex["ctxs"],
            "answers": ex["answers"]
        },
        num_proc=args.num_proc,
        desc="Generate prompt",
    )
    print(ds)

    # --------------------------------------------------------------------- #
    print("[+] Loading vLLM model ...")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.world_size,
        trust_remote_code=False,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    # --------------------------------------------------------------------- #
    out_path = Path(args.dest_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path / "generated_outputs_0_9600.jsonl"

    print("[+] Start generation ...")
    generation_start = time.time()
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(ds), args.batch_size), desc="vLLM batch infer"):
            batch = ds[i : i + args.batch_size]
            prompts = batch["prompt"]
            gens = llm.generate(prompts, SamplingParams(max_tokens=args.max_new_tokens))
            prompts    = batch["prompt"]     # list[str]
            questions  = batch["question"]   # list[str]
            ctxs_list  = batch["ctxs"] 
            answers = batch["answers"]

            for question, ctxs, gen, answer in zip(questions, ctxs_list, gens, answers):
                output_text = gen.outputs[0].text
                # print(f'output: {output_text}')

                triples = parse_triples(output_text)

                item = {
                    "question": question,
                    "passages": ctxs,
                    "triples" : triples,
                    "answers": answer,
                }
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    generation_end = time.time()
    print(f"[+] Generation completed and spent {generation_end-generation_start:.2f} seconds!")
    print(f"[✓] Saved structured triples to {jsonl_path}")