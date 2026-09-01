"""The Jinja2 subset chat templates are written in.

Not a general Jinja implementation and not trying to be. A chat template is the
model author's own artifact, which is what lets molla avoid maintaining a prompt
renderer per model family, and it is also untrusted input downloaded from the
internet. Both facts shape this package: the supported subset is pinned and
anything outside it is a named error at load time rather than a quiet
misrender, and rendering runs under a step budget, an output cap, a recursion
limit and a deadline.
"""
