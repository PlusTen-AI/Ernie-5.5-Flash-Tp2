"""The ERNIE chat protocol as the checkpoint ships it: encoding_ernie.py (template → segments) and decoding_ernie.py
(completion → reasoning / content / tool calls), copied unchanged (Apache-2.0, Baidu), plus tokens.py: the
segments → token ids step of the checkpoint's tokenization_ernie.py (ErnieChatTokenizer.encode_segments) on the
`tokenizers` library alone."""
