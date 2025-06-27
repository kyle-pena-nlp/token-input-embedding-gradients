import numpy as np
from IPython.display import HTML
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colormaps
import scipy.stats
from compute_batch_bert_gradients import BatchArgs
from bert_models import tokenizer
from compute_batch_bert_gradients import tokenizer
import matplotlib.ticker as mtick
from bert_util import clean_token, calculate_token_probabilities, invert_embeddings
from bert_models import MODEL


def viz_sentences_for_input_embed(args : BatchArgs, input_embeds : np.ndarray) -> list[list[str]]:
    input_ids = args.tokenization['input_ids']
    sentences = [ [ clean_token(token) for token in tokenizer.convert_ids_to_tokens(input_ids[i]) ] for i in range(len(args.tokenization['input_ids'])) ]
    masked_token_probabilities = calculate_token_probabilities(args, input_embeds)
    top_token_ids = np.argmax(masked_token_probabilities, axis=1)
    for i in range(len(sentences)):
        sentences[i][args.mask_positions[i]] = clean_token(tokenizer.convert_ids_to_tokens([top_token_ids[i]])[0])
    return sentences


def generate_saliency_html(args : BatchArgs, viz_sentences : list[list[str]], saliencies : list[np.ndarray], mask_token = None):

    html_output = []

    for sentence_idx, (sentence, sentence_saliencies, mask_position) in enumerate(zip(viz_sentences, saliencies, args.mask_positions)):

        sentence_html = '<div style="margin: 10px 0; font-family: monospace; line-height: 2.5;">'

        max_salience = np.max(sentence_saliencies) if len(sentence_saliencies) > 0 else 1.0

        for i, (token, salience) in enumerate(zip(sentence, sentence_saliencies)):
            if token in ('[CLS]', '[SEP]', '[PAD]'):
                continue

            if args.tokenization['input_ids'][sentence_idx][i] in (tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id):
                continue

            if i == mask_position:
                # Highlight the masked token with a different color
                sentence_html += f'<span style="background-color: #FFC107; padding: 3px; margin: 2px; border-radius: 3px; color: black">{mask_token or f"[{token}]"}</span>'
            else:
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


def viz_bert(args : BatchArgs, input_embeds_list):

    import torch
    from transformers import AutoModel
    from bertviz import model_view

    viz_model = AutoModel.from_pretrained(MODEL, output_attentions=True)  # Configure model to return attention values
    tokenized = args.tokenizations

    def viz_attention(inputs_embeds : torch.Tensor, tokenized, sentence_idx):
        # Omit input_ids from the tokenization so the model uses the input_embeds instead
        tokenized_without_input_ids = { key: value for (key,value) in tokenized.items() if key != "input_ids" }
        outputs = viz_model(**tokenized_without_input_ids, inputs_embeds=inputs_embeds)  # Run model
        attention = outputs[-1]

        sentences = invert_embeddings(inputs_embeds)[1]  # Convert input ids to token strings
        sentences = [ [ clean_token(token) for token in sentence ] for sentence in sentences ]
        tokens = sentences[sentence_idx]
        model_view(attention, tokens, include_layers=[0,1,2,3,25,26,27])  # Display model view

    viz_attention(torch.Tensor(input_embeds_list[0]), tokenized, 0)
    viz_attention(torch.Tensor(input_embeds_list[-1]), tokenized, 0)


    def viz_attention_diff(inputs_embeds_1 : torch.Tensor, inputs_embeds_2 : torch.Tensor, tokenized, sentence_idx):
        # Omit input_ids from the tokenization so the model uses the input_embeds instead
        tokenized_without_input_ids = { key: value for (key,value) in tokenized.items() if key != "input_ids" }
        outputs = viz_model(**tokenized_without_input_ids, inputs_embeds=inputs_embeds_1)  # Run model
        attention_1 = outputs[-1]

        tokenized_without_input_ids = { key: value for (key,value) in tokenized.items() if key != "input_ids" }
        outputs = viz_model(**tokenized_without_input_ids, inputs_embeds=inputs_embeds_2)  # Run model
        attention_2 = outputs[-1]

        attention = [ torch.clip(attn_1 - attn_2,0.) for (attn_1,attn_2) in zip(attention_1, attention_2) ]

        sentences = invert_embeddings(inputs_embeds_1)[1]  # Convert input ids to token strings
        sentences = [ [ clean_token(token) for token in sentence ] for sentence in sentences ]
        tokens = sentences[sentence_idx]
        model_view(attention, tokens, include_layers = [0,1,2,3,25,26,27])  # Display model view

    viz_attention_diff(torch.Tensor(input_embeds_list[0]), torch.Tensor(input_embeds_list[-1]), tokenized, 0)
    viz_attention_diff(torch.Tensor(input_embeds_list[-1]), torch.Tensor(input_embeds_list[0]), tokenized, 0)


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
