import multiprocessing as mp
import regex as re
from typing import List, Dict, Tuple
import heapq
import collections
import json
import pickle

from cs336_basics.pretokenization_example import find_chunk_boundaries

NUM_PROCESSES = 4


class ReverseLexOrderPair:
    """
    Encapsulates (bytes, bytes) so that in a min-heap, the "largest in normal lex order"
    is treated as the smallest. Ensures that tie frequencies pop in reverse lex order.
    """

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "ReverseLexOrderPair") -> bool:
        # Invert normal order: self < other if self is > other (so larger lex sorts first).
        return self.pair > other.pair

    def __eq__(self, other: "ReverseLexOrderPair") -> bool:
        return self.pair == other.pair

class BPETokenizerTrainer():
    def __init__(self, 
                 data_path: str,
                 vocab_size: int,
                 special_tokens: List[str],
                 chunk_special_token: str):
        self.data_path = data_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.chunk_special_token = chunk_special_token
        self.vocab = self.init_vocab()

    def remove_special_tokens(self, text: str):
        escaped_special_tokens = [re.escape(special_tok) for special_tok in self.special_tokens]
        split_pattern = "|".join(escaped_special_tokens)
        return re.split(split_pattern, text)
    
    def pretokenize(self, text: str) -> Dict[Tuple[bytes], int]:
        text_splitted = self.remove_special_tokens(text)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        pre_tokens = {}
        for text_chunk in text_splitted:
            for match in re.finditer(PAT, text_chunk):
                tok_bytes = tuple(bytes([b]) for b in match.group(0).encode("utf-8"))
                pre_tokens[tok_bytes] = pre_tokens.get(tok_bytes, 0) + 1

        return pre_tokens
    
    def process_chunk(self, start_end):
        start, end = start_end

        # Each process opens the file separately
        with open(self.data_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")

        pre_token_counts = self.pretokenize(chunk)

        return pre_token_counts

    def pretokenize_file(self) -> Dict[Tuple[bytes], int]:
        chunk_token_bytes: bytes = self.chunk_special_token.encode("utf-8")
        with open(self.data_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, NUM_PROCESSES, chunk_token_bytes)
            chunk_starts_ends = list(zip(boundaries[:-1], boundaries[1:]))

        # Parallel execution
        with mp.Pool(NUM_PROCESSES) as pool:
            results = pool.map(self.process_chunk, chunk_starts_ends)
            
        pre_token_counts = {}
        for res in results:
            for pre_tok in res:
                pre_token_counts[pre_tok] = pre_token_counts.get(pre_tok, 0) + res[pre_tok]
        return pre_token_counts
    
    def init_vocab(self):
        vocab = {i: bytes([i]) for i in range(256)}
        vocab.update(
            {i+256: (self.special_tokens[i]).encode("utf-8") for i in range(len(self.special_tokens))}
        )
        return vocab
    
    def get_new_pre_token(self,
                            old_pre_token,
                            pair):
        new_pre_token_bytes = []
        i = 0
        while i < len(old_pre_token):
            if i < len(old_pre_token)-1 and (old_pre_token[i] == pair[0] and old_pre_token[i+1] == pair[1]):   # Merge the pair in the pre-token: b1+b2
                new_pre_token_bytes.append(pair[0] + pair[1])
                i += 2
            else:
                new_pre_token_bytes.append(old_pre_token[i])
                i += 1
        return tuple(new_pre_token_bytes)

    def merge_pair(self, 
                   pair: Tuple[bytes],
                   pre_token_counts: Dict[Tuple[bytes], int],
                   pair_counts: Dict[Tuple[bytes], int],
                   pair_heap: List,
                   pair_to_pre_token: Dict[Tuple[bytes], set]):
        pre_tokens_to_modify = pair_to_pre_token[pair].copy()

        changed_pairs = set()
        for old_tok in pre_tokens_to_modify:
            old_tok_count = pre_token_counts.pop(old_tok)

            # Update pre-tokens
            new_tok = self.get_new_pre_token(old_tok, pair)
            pre_token_counts[new_tok] = pre_token_counts.get(new_tok, 0) + old_tok_count

            # Decrement pair-counts of pairs in old pre-token adjacencies, record the changed_pairs, 
            # modify the pair_to_pre_token
            for i in range(len(old_tok) - 1):
                old_pair = (old_tok[i], old_tok[i+1])
                pair_counts[old_pair] -= old_tok_count
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
                changed_pairs.add(old_pair)
                pair_to_pre_token[old_pair].discard(old_tok)

            # Increment pair-counts of pairs in new pre-token adjacencies, record the changed_pairs, 
            # modify the pair_to_pre_token
            for i in range(len(new_tok) - 1):
                new_pair = (new_tok[i], new_tok[i+1])
                pair_counts[new_pair] = pair_counts.get(new_pair, 0) + old_tok_count
                changed_pairs.add(new_pair)
                pair_to_pre_token[new_pair].add(new_tok)

        # Empty the pre_token sets of the target pair
        pair_to_pre_token[pair] = set()

        # Push the pair-counts of the changed_pairs to the heap
        for cp in changed_pairs:
            if cp in pair_counts and pair_counts[cp] > 0:
                heapq.heappush(pair_heap, (-pair_counts[cp], ReverseLexOrderPair(cp), cp))


    def train(self) -> Tuple[Dict[int,bytes], List[tuple[bytes, bytes]]]:
        # Pre-tokenization
        pre_token_counts = self.pretokenize_file()

        # Init byte-pair counts
        pair_counts = {}
        pair_to_pre_token = collections.defaultdict(set)
        for pre_tok, pre_tok_count in pre_token_counts.items():
            for ind in range(len(pre_tok)-1):
                pair = (pre_tok[ind], pre_tok[ind+1])
                pair_counts[pair] = pair_counts.get(pair, 0) + pre_tok_count
                pair_to_pre_token[pair].add(pre_tok)

        # Build a pair-heap by pushing negative pair counts
        pair_heap = []
        for pair, count in pair_counts.items():
            heapq.heappush(pair_heap, (-count, ReverseLexOrderPair(pair), pair))

        # Training process (merge pairs until reaching the vocab size/ no pairs to merge)
        initial_vocab_size = len(self.vocab)

        merges = []
        for i in range(initial_vocab_size, self.vocab_size):
            # Select top-pair: Pop the pair-heap until we find the pair which still matches the pair_counts
            while pair_heap:
                neg_count, _, top_pair = heapq.heappop(pair_heap)
                if top_pair in pair_counts and pair_counts[top_pair] == -neg_count:
                    break
            else:
                break

            # Add the pair to the vocab
            self.vocab[i] = top_pair[0] + top_pair[1]
            merges.append(top_pair)

            # Apply the merge
            self.merge_pair(
                top_pair,
                pre_token_counts,
                pair_counts,
                pair_heap,
                pair_to_pre_token
            )

        return (self.vocab, merges)


if __name__ == "__main__":
    chunk_token = "<|endoftext|>"
    vocab_size = 2000
    special_tokens = ["<|endoftext|>"]
    file_path = r"E:\LLM\CS336\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt"

    bpe_trainer = BPETokenizerTrainer(
        data_path=file_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        chunk_special_token=chunk_token
    )

    vocab, merges = bpe_trainer.train()
    # # print(vocab.values())
    # # print(len(vocab))
    # # print(len(merges))

    # # Save the vocab
    # with open(r"cs336_basics\vocab.pkl", "wb") as f:
    #     pickle.dump(vocab, f)