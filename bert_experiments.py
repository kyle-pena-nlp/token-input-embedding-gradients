
from compute_batch_bert_gradients import BatchArgs
from bert_util import get_token_distribution

HORSE_DISTRIBUTION = get_token_distribution(["The animal that says neigh is a [MASK]"])

experiments = {
    "one-step": BatchArgs(steps = 1,
        num_examples = 1,
        examples_filepath='example_sentences.tsv',
        example_stride=1,
        scramble_target_probs=False,
        target_probabilities = HORSE_DISTRIBUTION,
        randomize_input_embeds = False),

    "horse": BatchArgs(steps = 250,
        num_examples = 1,
        examples_filepath='example_sentences.tsv',
        example_stride=1,
        target_probabilities=HORSE_DISTRIBUTION),

    "eng": BatchArgs(steps = 250,
        num_examples = 64,
        examples_filepath='eng_sentences.tsv',
        example_stride=50),

    "eng-random-embedding": BatchArgs(steps = 250,
        num_examples = 64,
        examples_filepath='eng_sentences.tsv',
        example_stride=50,
        randomize_input_embeds = True)
}
