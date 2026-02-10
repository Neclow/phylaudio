import torch
from tqdm import tqdm

from src._config import DEFAULT_EMBEDDING_DIR, MIN_LANGUAGES
from src.tasks.feature_extraction.base import (
    get_fleurs_parallel_args,
    prepare_everything,
    save_state,
    sentence_loop,
)

if __name__ == "__main__":
    args = get_fleurs_parallel_args().parse_args()

    inputs = prepare_everything(args)
    dataset_embedding_dir = f"{DEFAULT_EMBEDDING_DIR}/{args.dataset}"
    output_folder = f"{dataset_embedding_dir}/{inputs.run_id}"

    save_state(inputs, output_folder)

    all_y = []

    with torch.no_grad():
        for batch in tqdm(
            inputs.parallel_loader,
            total=inputs.num_batches,
            desc="(base) Processing sentence data",
        ):
            X_input = batch["input"]
            y = batch["label"][0].to(args.device)
            attention_mask = batch["attention_mask"][0].to(args.device)
            sentence_index = batch["sentence_index"][0]

            # Ignore if less than 4 languages ==> cannot build a tree
            if y.unique().shape[0] < MIN_LANGUAGES:
                continue

            all_y.append(y.cpu())

        y = torch.cat(all_y)

    torch.save(y, f"{output_folder}/labels.pt")

    all_X_emb = []

    sentence_loop(
        args,
        inputs,
        output_folder=None,
        downstream_func=lambda x, *args: all_X_emb.append(x),
    )

    X_emb_cat = torch.cat(all_X_emb, dim=0)

    torch.save(X_emb_cat, f"{output_folder}/embeddings.pt")
