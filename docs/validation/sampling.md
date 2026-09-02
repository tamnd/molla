# Sampling

Issue #28. [decode.md](decode.md) gets a row of logits out of a model. This is the part that turns that row into one token, and it is the part users can see from the outside without knowing what any of it is called.

## The order is the definition

A sampler is a pipeline of filters over a candidate set, and the order they run in is not an implementation detail. Top-p after top-k measures the mass over what top-k left. The same two the other way round is a different sampler, and people will report the difference as a bug.

The order here is llama.cpp's:

    grammar mask, logit bias, penalties, temperature, top-k, top-p, min-p, typical, sample

That is the order every preset in circulation was tuned against. A preset copied out of somebody's config with a temperature of 0.7 and a top-p of 0.9 was arrived at by trying numbers until the output read well, against a program that ran the filters in that sequence, so running them in a different one gives a different model to the person who brought the preset.

Grammar masking is M4. The hook is here and does nothing, which is a placeholder rather than an omission: it has to run first, so that a token the grammar forbids is gone before anything measures how much probability mass is left. A grammar applied after top-p is a grammar applied to a set that was truncated on the assumption every token was allowed, and that produces a sampler which occasionally has nothing left to choose from.

A logit bias is added rather than multiplied, so the same number means the same thing at every temperature and a bias of minus one hundred is a ban. It goes in after the mask so a bias cannot bring back a token the grammar forbade, and before the penalties so that a caller pushing a token down and the penalties pushing the same token down compose rather than one of them winning.

## Greedy is the argmax, not a small temperature

A temperature of a thousandth is not greedy decoding. It divides logits that already differ by tens, and the exponential of the result overflows to infinity and comes back as a NaN, so the limit that is supposed to be greedy is instead undefined. Greedy here means take the largest and do not build a distribution at all.

That path also skips the candidate copy, because it is exactly the argmax over the penalised logits and nothing else. Temperature and every filter here are monotonic, so none of them can move which token is largest. Penalties can, and biases can, so the shortcut is only taken when neither is in play.

## The randomness is a counter, not a stream

A stream sampler draws its next float from wherever the last draw left the state. Two sequences sharing a generator then get tokens that depend on the interleaving, and the same seed with the same prompt gives different output depending on what else was in the batch. That is not reproducibility, it is reproducibility as long as nobody else is using the server, which is the version that holds in testing and fails in production.

So the generator hashes the seed together with the draw number. Draw `n` of a sequence is a pure function of those two, the batch cannot be observed from inside it, and a sequence that is preempted and resumed picks up exactly where it was. The mixing function is splitmix64's finalizer, applied to a counter rather than iterated on a state.

The float is twenty four bits over two to the twenty four. That is every value a float32 can hold in `[0, 1)` without rounding two counters onto the same float, and taking sixty four bits and dividing would round anyway while hiding that it had.

## Logprobs are the model's numbers

A client asking for logprobs is asking what the model thought, not what the sampler did with it. So they come from the logits as they arrived, before the penalties and before the temperature. Computed the other way, two clients running the same model on the same prompt at different temperatures would get different logprobs and would reasonably call that a bug.

They are computed as `z - max - log(sum(exp(z - max)))` rather than as the log of a softmax, so a token whose probability is below what a float32 can hold comes back as a large negative number instead of minus infinity.

## What is checked

Fifty two checks, in `tests/test_sample.mojo`. The logits under test are the log of a half, a quarter, an eighth and two sixteenths, so the softmax at temperature one is exact powers of two and every threshold below is a number a person can check rather than one the test computed.

Most of it samples a few hundred times and asks which tokens were reachable, because a filter that is off by one candidate still produces a token that was always allowed. One call cannot tell the difference and four hundred can.

The done criterion from #28 is in there directly: twenty tokens, a hundred fresh samplers on the same seed, all twenty identical every time. Next to it is the property that motivated the counter, which is a second sequence drawing in between leaving the first sequence's tokens untouched.

The rest is one check per boundary that a plausible implementation gets wrong. That top-p keeps the token which carries the sum past the line rather than dropping it. That min-p cuts at a fraction of the most likely token rather than at an absolute probability, so it keeps almost nothing from a peaked distribution and almost everything from a flat one. That a repetition penalty divides a positive logit and multiplies a negative one, checked by penalising every token and asking that the only negative one is still last, since doing one operation to both signs rewards half the vocabulary for having just been used. That a penalty window of zero turns all three penalties off, and that the window forgets what fell off the back of it. That logprobs sum to one, are unchanged by adding five hundred to every logit, and show neither the temperature nor the penalties that were set on the sampler that produced them.

## From the command line

`molla generate` takes the settings as named flags, in the same long form with an equals sign the configuration flags use:

    molla generate model.gguf tokenizer.json "Once upon a time" 40 512 \
      --temp=0.9 --top-k=40 --top-p=0.95 --min-p=0.05 --repeat-penalty=1.1 --seed=7

The two numbers stay positional because they were already in scripts, and everything else is named because nobody is going to remember a ninth position. An unknown flag is an error rather than something silently ignored, and the settings are checked before the file is opened, so a typo in a top-p is reported immediately instead of once an 8B has finished loading.

No flags means greedy. That matters for the same reason the default matters everywhere else here: a run that reads badly should be the model or the kernels, not a draw that went somewhere unlikely.

The settings that were used are printed with the model and the context, because a run that reads oddly is the first thing anybody argues about and the argument is shorter when the numbers behind it are in the same output as the text.

## What this is not

There is no grammar. The hook is in the right place and returns immediately, and constrained decoding is M4.

There is no logit bias flag on the command line. The bias lives on the sampler and is set per request, which is what the OpenAI `logit_bias` field is, and that field arrives with the routes in #29.

Nothing here is fused or on a device. The penalties are a loop over the candidate set on the host, and the sort is a heapsort rather than a partial selection, which is the right shape and not the right speed. Heapsort rather than quicksort because a row of logits is not random data and the worst case of a bad pivot over a hundred thousand candidates is a visible stall on one token in a conversation, which is exactly the kind of intermittent slowness nobody ever tracks down.

Mirostat is not here and is not planned for M2. It is a feedback controller over the surprise of what has already been generated, which is a different shape from every filter above, and adding it as a tenth stage of the same pipeline is how it usually gets implemented wrong.
