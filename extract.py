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
    """将大模型生成的文本解析为结构化 triples 列表"""
    triples = []
    lines = raw_text.strip().split(tuple_delimiter[0])  # 快速切分
    raw_text = raw_text.strip()

    if completion_delimiter in raw_text:
        raw_text = raw_text.split(completion_delimiter)[0]

    triple_pattern = re.compile(r'\("(?P<type>entity|relationship)"\|(?P<fields>.*?)\)')

    for match in triple_pattern.finditer(raw_text):
        ttype = match.group("type")
        fields = match.group("fields").split(tuple_delimiter)

        if ttype == "entity" and len(fields) == 3:
            triples.append({
                "type": "entity",
                "name": fields[0].strip('" '),
                "entity_type": fields[1].strip('" '),
                "desc": fields[2].strip('" ')
            })
        elif ttype == "relationship" and len(fields) == 4:
            triples.append({
                "type": "relation",
                "head": fields[0].strip('" '),
                "tail": fields[1].strip('" '),
                "desc": fields[2].strip('" '),
                "score": int(fields[3]) if fields[3].isdigit() else 5
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
    ds = load_dataset("json", data_files=args.data_file, split="train")
    print(f"    Dataset size = {len(ds)}")

    print("[+] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # 预生成 prompt 字符串
    print("[+] Building prompts in parallel ...")
    ds = ds.map(
        lambda ex: {"prompt": build_prompt(ex, tokenizer)},
        num_proc=args.num_proc,
        desc="Generate prompt",
    )

    # --------------------------------------------------------------------- #
    print("[+] Loading vLLM model ...")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.world_size,
        trust_remote_code=True,
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
    jsonl_path = out_path / "generated_outputs.jsonl"

    print("[+] Start generation ...")
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(ds), args.batch_size), desc="vLLM batch infer"):
            batch = ds[i : i + args.batch_size]
            prompts = batch["prompt"]
            gens = model.generate(prompts, SamplingParams(max_tokens=args.max_new_tokens))

            for ex, gen in zip(batch, gens):
                output_text = gen.outputs[0].text
                triples = parse_triples(output_text)

                item = {
                    "question": ex.get("question", None),
                    "passages": ex.get("ctxs", []),  # title + text
                    "triples": triples
                }

                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[✓] Saved structured triples to {jsonl_path}")