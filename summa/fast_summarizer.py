from math import log10

from .pagerank_weighted import pagerank_weighted_scipy as _pagerank
from .preprocessing.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from .commons import build_graph as _build_graph
from .commons import remove_unreachable_nodes as _remove_unreachable_nodes
from .embeddings import get_embedding_similarity


import editdistance
import io
import itertools
import networkx as nx
import nltk
import os
from sacremoses import MosesTokenizer
from collections import OrderedDict


def filter_for_tags(tagged, tags=['NN', 'JJ', 'NNP']):
    """Apply syntactic filters based on POS tags."""
    return [item for item in tagged if item[1] in tags]


def normalize(tagged):
    """Return a list of tuples with the first item's periods removed."""
    return [(item[0].replace('.', ''), item[1]) for item in tagged]


def unique_everseen(iterable, key=None):
    """List unique elements in order of appearance.

    Examples:
        unique_everseen('AAAABBBCCDAABBB') --> A B C D
        unique_everseen('ABBCcAD', str.lower) --> A B C D
    """
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in [x for x in iterable if x not in seen]:
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

def count_common_words(words_sentence_one, words_sentence_two):
    return len(set(words_sentence_one) & set(words_sentence_two))


def build_graph(nodes, weight_function):
    """Return a networkx graph instance.

    :param nodes: List of hashables that represent the nodes of a graph.
    """
    gr = nx.Graph()  # initialize an undirected graph
    gr.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))

    # add edges to the graph (weighted by Levenshtein distance)
    for pair in nodePairs:
        firstString = pair[0]
        secondString = pair[1]
        edge_weight = weight_function(firstString, secondString)
        #levDistance = editdistance.eval(firstString, secondString)
        gr.add_edge(firstString, secondString, weight=edge_weight)

    return gr


def edit_distance(sentence_1, sentence_2):
    return editdistance.eval(sentence_1, sentence_2)

def lexical_overlap(sentence_1, sentence_2):
    words_sentence_one = sentence_1.split()
    words_sentence_two = sentence_2.split()
    common_word_count = count_common_words(words_sentence_one, words_sentence_two)

    log_s1 = log10(len(words_sentence_one))
    log_s2 = log10(len(words_sentence_two))

    if log_s1 + log_s2 == 0:
        return 0
    return common_word_count / (log_s1 + log_s2)


def embeddding_similarity(sentence_1, sentence_2):
    return get_embedding_similarity(sentence_1, sentence_2)


def extract_key_phrases(text):
    """Return a set of key phrases.

    :param text: A string.
    """
    # tokenize the text using nltk
    tokenizer = MosesTokenizer(lang="en")
    word_tokens = tokenizer.tokenize(text)

    # assign POS tags to the words in the text
    tagged = nltk.pos_tag(word_tokens)
    textlist = [x[0] for x in tagged]

    tagged = filter_for_tags(tagged)
    tagged = normalize(tagged)

    unique_word_set = unique_everseen([x[0] for x in tagged])
    word_set_list = list(unique_word_set)

    # this will be used to determine adjacent words in order to construct
    # keyphrases with two words

    graph = build_graph(word_set_list)

    # pageRank - initial value of 1.0, error tolerance of 0,0001,
    calculated_page_rank = nx.pagerank(graph, weight='weight')

    # most important words in ascending order of importance
    keyphrases = sorted(calculated_page_rank, key=calculated_page_rank.get,
                        reverse=True)

    # the number of keyphrases returned will be relative to the size of the
    # text (a third of the number of vertices)
    one_third = len(word_set_list) // 3
    keyphrases = keyphrases[0:one_third + 1]

    # take keyphrases with multiple words into consideration as done in the
    # paper - if two words are adjacent in the text and are selected as
    # keywords, join them together
    modified_key_phrases = OrderedDict()
    # keeps track of individual keywords that have been joined to form a
    # keyphrase
    dealt_with = OrderedDict()
    i = 0
    j = 1
    while j < len(textlist):
        first = textlist[i]
        second = textlist[j]
        if first in keyphrases and second in keyphrases:
            keyphrase = first + ' ' + second
            modified_key_phrases[keyphrase] = None
            dealt_with[first] = None
            dealt_with[second] = None
        else:
            if first in keyphrases and first not in dealt_with:
                modified_key_phrases[first] = None

            # if this is the last word in the text, and it is a keyword, it
            # definitely has no chance of being a keyphrase at this point
            if j == len(textlist) - 1 and second in keyphrases and \
                    second not in dealt_with:
                modified_key_phrases[second] = None

        i = i + 1
        j = j + 1

    return list(modified_key_phrases.keys())


def summarize(text,ratio=0.2, language='english', clean_sentences=False, weight_function="edit_distance"):
    """Return a paragraph formatted summary of the source text.

    :param text: A string.
    """
    sent_detector = nltk.data.load('tokenizers/punkt/'+language+'.pickle')
    sentence_tokens = sent_detector.tokenize(text.strip())
    if weight_function == "edit_distance":
        graph = build_graph(sentence_tokens, edit_distance)
    if weight_function == "lexical_overlap":
        graph = build_graph(sentence_tokens, lexical_overlap)
    if weight_function == "embedding_similarity":
        graph  = build_graph(sentence_tokens, embeddding_similarity)
    
    calculated_page_rank = nx.pagerank(graph, weight='weight')

    # most important sentences in ascending order of importance
    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get,
                       reverse=True)
    # calculate ratio of important sentences to be returned as summary
    length = len(sentences) * ratio
    _temp_sentences = sentences[:int(length)]
    _summary_sentences = [(i,item) for i, item in enumerate(_temp_sentences)]
    _summary_sentences.sort(key=lambda x: sentence_tokens.index(x[1]))
    summary_sentences = [item for i, item in _summary_sentences]
    return ' '.join(summary_sentences)

def write_files(summary, key_phrases, filename):
    """Write key phrases and summaries to a file."""
    print("Generating output to " + 'keywords/' + filename)
    key_phrase_file = io.open('keywords/' + filename, 'w')
    for key_phrase in key_phrases:
        key_phrase_file.write(key_phrase + '\n')
    key_phrase_file.close()

    print("Generating output to " + 'summaries/' + filename)
    summary_file = io.open('summaries/' + filename, 'w')
    summary_file.write(summary)
    summary_file.close()

    print("-")


def summarize_all():
    # retrieve each of the articles
    articles = os.listdir("articles")
    for article in articles:
        print('Reading articles/' + article)
        article_file = io.open('articles/' + article, 'r')
        text = article_file.read()
        keyphrases = extract_key_phrases(text)
        summary = summarize(text)
        write_files(summary, keyphrases, article)
