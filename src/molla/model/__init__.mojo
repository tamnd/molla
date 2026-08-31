"""Reading model files.

The model plane is everything between a file on disk and a tensor the engine
can hand to a kernel. It starts with GGUF because that is what most local
weights are distributed as, and it starts with metadata because knowing what a
file claims to be has to work before reading what is in it means anything.

Nothing here loads weights. Files are mapped and walked in place, so opening a
4 GB model costs no heap and no copy.
"""
