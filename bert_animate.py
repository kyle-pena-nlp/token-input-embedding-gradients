import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.stats
from compute_batch_bert_gradients import BatchArgs
from bert_util import clean_token, invert_embeddings
from bert_models import tokenizer

def animate_kl_divergences(args : BatchArgs, probs_path : list[np.ndarray]):
    # Plot KL-Divergence over time
    max_anim_steps = 250
    kl_divergence = scipy.stats.entropy(probs_path, np.expand_dims(args.target_probabilities,0), axis = 2)
    frames = min(len(kl_divergence), max_anim_steps)
    xs = np.arange(len(kl_divergence)) # (500)
    ys = kl_divergence # (500,64)

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.set_tight_layout(True)
    lines = [ax.plot([], [], color="blue", alpha=0.1)[0] for i in range(ys.shape[1])]

    def init():
        ax.set_xlim(0, len(xs))
        ax.set_ylim(-0.01, np.max(ys) * 1.1)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('KL Divergence')
        ax.set_title('KL Divergences')
        ax.grid(True, which='both')
        return lines

    def animate(frame):
        for i, line in enumerate(lines):
            index = frame * int(len(xs)/frames)
            line.set_data(xs[:index], ys[:index, i])
        return lines

    ani = FuncAnimation(fig, animate, frames=frames, init_func=init, blit=True, interval=50)
    plt.close()  # Prevents duplicate display in Jupyter
    return HTML(ani.to_jshtml())


def animate_sentence_level_L2_distances(args : BatchArgs, input_embeds_list : list[np.ndarray]):

    max_anim_steps = 250
    np_inputs_embeds_list = np.asarray(input_embeds_list)
    start_inputs_embeds = np.expand_dims(input_embeds_list[0], axis = 0)
    input_embeds_diffs = start_inputs_embeds - np_inputs_embeds_list
    distances_from_origins = np.linalg.norm(input_embeds_diffs.reshape(*input_embeds_diffs.shape[:-2], -1), axis = 2, ord = 2)
    distances_from_origins.shape # (500,64)
    frames = min(max_anim_steps,distances_from_origins.shape[0])


    fig, ax = plt.subplots(figsize=(5, 3))
    fig.set_tight_layout(True)
    lines = [ax.plot([], [], lw=1, color = "blue", alpha = 0.1)[0] for _ in range(distances_from_origins.shape[1])]
    ax.set_xlim(0, distances_from_origins.shape[0])
    ax.set_ylim(0, distances_from_origins.max() * 1.1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('L2 Distance')
    ax.set_title('L2 Distances From Initial Input Embedding (64 Examples)')

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(i):
        index = i * int(len(distances_from_origins)/frames)
        x = range(index+1)
        for j, line in enumerate(lines):
            line.set_data(x, distances_from_origins[:index+1, j])
        return lines

    ani = FuncAnimation(fig, animate, frames=frames,
                        init_func=init, blit=True, interval=50)
    plt.close()  # Prevents duplicate display in Jupyter
    return HTML(ani.to_jshtml())


def animate_token_level_L2_distances(args : BatchArgs, input_embeds_list : list[np.ndarray], sentence_idx : int):

    max_anim_steps = 250
    np_inputs_embeds_list = np.asarray([x[sentence_idx,...] for x in input_embeds_list]) # [t,N,V]
    start_inputs_embeds = input_embeds_list[0] # [1,N,V]
    input_embeds_diffs = start_inputs_embeds - np_inputs_embeds_list # [t,N,V]
    distances_from_origins = np.linalg.norm(input_embeds_diffs, axis = 2, ord = 2) # [t,N]
    frames = min(max_anim_steps,distances_from_origins.shape[0])

    fig, ax = plt.subplots(figsize=(7, 4))
    lines = [ax.plot([], [], lw=1, color = "blue", alpha = 0.1)[0] for _ in range(distances_from_origins.shape[1])]
    texts = [ax.text(0, distances_from_origins[0,i], "", fontsize=8) for i in range(distances_from_origins.shape[1])]
    ax.set_xlim(0, distances_from_origins.shape[0])
    ax.set_ylim(0, distances_from_origins.max() * 1.1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('L2 Distance from Initial Input Embedding')
    ax.set_title('L2 Distance From Initial Input Embedding')

    def clean_token(token):
        if token in ["[CLS]","[SEP]","[MASK]"]:
            return ""
        return token.replace("Ġ", " ")

    def init():
        for line in lines:
            line.set_data([], [])
        for text in texts:
            text.set_position((0, 0))
            text.set_text("")
        return lines + texts

    def animate(i):
        index = i * int(len(distances_from_origins)/frames)

        nearest_ids, nearest_tokens = invert_embeddings(input_embeds_list[index])
        nearest_tokens = [clean_token(token) for token in nearest_tokens[sentence_idx]]

        x = range(index+1)
        for j, (line, text) in enumerate(zip(lines, texts)):
            line.set_data(x, distances_from_origins[:index+1, j])
            # Update text position and content
            if index > 0:  # Only update text if we have points plotted
                text.set_position((index, distances_from_origins[index, j]))
                text.set_text(nearest_tokens[j])
        return lines + texts

    ani = FuncAnimation(fig, animate, frames=frames,
                        init_func=init, blit=True, interval=50)
    plt.close()  # Prevents duplicate display in Jupyter
    return HTML(ani.to_jshtml())

def animate_prob_distr_path(args : BatchArgs, input_embeds_list : list[np.ndarray], probs_path : list[np.ndarray], SELECTED_IDX : int = 0):
    special_tokens = set(['[CLS]', '[SEP]', '[PAD]'])
    def clean_sentence(sentence):
        string = ""
        for token in sentence:
            if token in special_tokens:
                continue
            if token == '[MASK]':
                token = ' ___'
            token = token.replace("Ġ", " ")
            string += token
        return string

    start_inputs_embeds = input_embeds_list[0]
    start_sentences = [ clean_sentence(sentence) for sentence in invert_embeddings(start_inputs_embeds)[1] ]
    final_inputs_embeds = input_embeds_list[-1]
    end_sentences = [ clean_sentence(sentence) for sentence in invert_embeddings(final_inputs_embeds)[1] ]


    target_probs = args.target_probabilities # [N,V]
    top_target_prob_idxs = np.argsort(-target_probs, axis = -1)[:,:10]
    top_target_tokens = [ clean_token(token) for token in  tokenizer.convert_ids_to_tokens(top_target_prob_idxs[SELECTED_IDX,:]) ]
    top_target_probs = np.take_along_axis(target_probs, top_target_prob_idxs, axis = -1)

    np_mask_positions = np.asarray(args.mask_positions)


    masked_token_indexed_prob_paths = np.take_along_axis(probs_path, np.expand_dims(top_target_prob_idxs, axis = 0), axis = 2)
    selected_masked_token_indexed_prob_paths = masked_token_indexed_prob_paths[:,SELECTED_IDX,:]
    selected_top_target_probs = top_target_probs[SELECTED_IDX,:]
    selected_source_sentence = start_sentences[SELECTED_IDX]
    selected_target_sentence = "The animal that says neigh is is a [ horse]"#start_sentences[args.sentence_permutation[SELECTED_IDX]]


    selected_sentence_top_token_path = np.argsort(-probs_path[:,SELECTED_IDX,:], axis=1)[:,0]
    selected_sentence_top_token_path = [ clean_token(token) for token in tokenizer.convert_ids_to_tokens(selected_sentence_top_token_path)]


    sentences_path = [ clean_sentence(invert_embeddings(input_embeds_list[i][[SELECTED_IDX],...])[1][0]) for i in range(len(input_embeds_list)) ]

    selected_top_target_token = clean_token(tokenizer.convert_ids_to_tokens(top_target_prob_idxs[SELECTED_IDX,:1])[0])

    fig, ax = plt.subplots(figsize=(8, 4))

    def init():
        ax.clear()
        return []

    def animate(i):
        ax.clear()
        probs = selected_masked_token_indexed_prob_paths[i]
        positions = range(len(probs))
        ax.bar(positions, selected_top_target_probs, color='gray', alpha = 0.25)
        bars = ax.bar(positions, probs, color='skyblue')
        ax.set_ylim(0, selected_masked_token_indexed_prob_paths.max() * 1.1)
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Probability')
        # Token Probabilities at Step {i} \n
        ax.set_title(f'"{sentences_path[i].replace("___", f"[{selected_sentence_top_token_path[i]}]")}"')
        # \nTarget: "{selected_target_sentence.replace("___", f"[{selected_top_target_token}]")}"')
        ax.set_xticks(range(len(top_target_tokens)))
        ax.set_xticklabels(labels=top_target_tokens)
        return bars

    ani = FuncAnimation(fig, animate, frames=selected_masked_token_indexed_prob_paths.shape[0],
                    init_func=init, blit=True, interval=50)
    plt.close()  # Prevents duplicate display in Jupyter
    HTML(ani.to_jshtml())
