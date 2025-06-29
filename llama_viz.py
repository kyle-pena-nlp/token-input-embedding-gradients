import numpy as np
from IPython.display import HTML
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colormaps
import scipy.stats
from compute_batch_llama_gradients import BatchArgs
from llama_util import tokenizer, clean_token, calculate_token_probabilities, invert_embeddings
import matplotlib.ticker as mtick


def viz_sentences_for_input_embed(args : BatchArgs, input_embeds : np.ndarray) -> list[list[str]]:
    input_ids = args.tokenization['input_ids']
    sentences = [ [ clean_token(token) for token in tokenizer.convert_ids_to_tokens(input_ids[i]) ] for i in range(len(args.tokenization['input_ids'])) ]
    next_token_probabilities = calculate_token_probabilities(args, input_embeds)
    top_token_ids = np.argmax(next_token_probabilities, axis=1)
    return sentences


def generate_saliency_html(args : BatchArgs, viz_sentences : list[list[str]], saliencies : list[np.ndarray]):

    html_output = []

    for sentence_idx, (sentence, sentence_saliencies) in enumerate(zip(viz_sentences, saliencies)):

        sentence_html = '<div style="margin: 10px 0; font-family: monospace; line-height: 2.5;">'

        max_salience = np.max(sentence_saliencies) if len(sentence_saliencies) > 0 else 1.0

        for i, (token, salience) in enumerate(zip(sentence, sentence_saliencies)):
            #if token in ('<|begin_of_text|>'):
            #    continue

            if args.tokenization['input_ids'][sentence_idx][i] in tokenizer.all_special_ids:
                continue

            # Normalize salience value
            intensity = min(1.0, max(0.0, salience / max_salience))
            # Convert to RGB - from white to deep blue
            r = int(255 * (1 - intensity))
            g = int(255 * (1 - intensity))
            b = 255

            display_token = clean_token(token)
            sentence_html += f'<span style="background-color: rgb({r},{g},{b}); padding: 3px; margin: 2px; border-radius: 3px; color: {"white" if intensity > 0.5 else "black"};">{display_token}</span>'

        sentence_html += '</div>'
        html_output.append('<div style="background-color:white">' + sentence_html + '</div>')

    return HTML(''.join(html_output))




def display_gradient_displacement(args : BatchArgs, input_embeds_list : list[np.ndarray], sentence_idx : int):

    last_input_embeds = input_embeds_list[-1][sentence_idx]
    first_input_embeds = input_embeds_list[0][sentence_idx]


    input_embeds_diff = last_input_embeds - first_input_embeds
    sentence_tokens = invert_embeddings(input_embeds_list[0])[1][sentence_idx]
    fig,ax = plt.subplots(1,1)
    ax2 = ax.twinx()
    ax2.set_ylim(0,1)
    cmap = colormaps['Set2']
    for i,token_diffs in enumerate(last_input_embeds):
        token_id =args.tokenization['input_ids'][sentence_idx][i]
        if token_id in tokenizer.all_special_ids:
            continue
        cleaned_token = clean_token(sentence_tokens[i])
        if cleaned_token == "":
            continue
        if args.gradient_masks[sentence_idx,i] == 0:
            continue

        ax.hist(token_diffs, alpha = 0.2, bins = 50, color=cmap(i))
        ax.set_xlabel("Displacement Of Dimension")
        ax.set_ylabel("Count")

        values, base = np.histogram(token_diffs, bins=50)
        cumulative = np.cumsum(values)
        cumulative = cumulative / cumulative[-1]

        ax2.plot(base[:-1], cumulative, c=cmap(i), label = cleaned_token)

    plt.suptitle("Histogram of Dimension Displacement With CDF")
    plt.legend()
    plt.show()



def plot_distribution(target_probs, SELECTED_IDX : int = 0, TOP_N : int = 6, ax = None):
    top_target_prob_idxs = np.argsort(-target_probs, axis = -1)[:,:TOP_N]
    top_target_probs = np.take_along_axis(target_probs, top_target_prob_idxs, axis = -1)
    selected_top_target_probs = top_target_probs[SELECTED_IDX,:TOP_N]
    top_target_tokens = [clean_token(token) for token in tokenizer.convert_ids_to_tokens(top_target_prob_idxs[SELECTED_IDX,:TOP_N])]
    if ax is None:
        fig, ax = plt.subplots()
    ax.bar(np.arange(TOP_N), selected_top_target_probs, alpha = 0.25)
    ax.set_xticks(np.arange(TOP_N))
    ax.set_xticklabels(labels=top_target_tokens)
    ax.set_title("The animal that says bark is a ____")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))


def viz_sentence_changes(args : BatchArgs, input_embeds_list : list[np.ndarray]):
    from llama_util import invert_embeddings, clean_token
    import difflib
    _,start_sentences = invert_embeddings(input_embeds_list[0])
    _,end_sentences = invert_embeddings(input_embeds_list[-1])
    html_output = []
    for (s1, s2) in zip(start_sentences, end_sentences):
        start_text = "".join([clean_token(t) for t in s1 if t != '<|end_of_text|>'])
        end_text = "".join([clean_token(t) for t in s2 if t != '<|end_of_text|>'])

        if True:
            html_output.append(f'<div style="background-color: #f8f8f8; padding: 10px; margin: 5px; border: 1px solid #ddd;">')

            # Generate inline diff
            diff = difflib.ndiff(start_text, end_text)
            diff_html = []

            for i, d in enumerate(diff):
                if d.startswith('- '):
                    diff_html.append(f'<span style="color: #a00; text-decoration: line-through;">{d[2:]}</span>')
                elif d.startswith('+ '):
                    diff_html.append(f'<span style="color: #0a0;">{d[2:]}</span>')
                elif d.startswith('  '):
                    diff_html.append(d[2:])

            html_output.append(''.join(diff_html))
            html_output.append(f'</div>')

    return HTML(''.join(html_output))
