# Katia Huri	Evyatar Iluz	307344861	26540690 #
from __future__ import division
import numpy as np
import sys
import math
from collections import Counter
import time
import matplotlib.pyplot as plt


# def main(argv):
def main(dev_file, topics):
    # ex3_process(argv[1], argv[2], argv[3],argv[4], argv[5])
    start = time.time()
    ex3_process(dev_file, topics)
    end = time.time()
    print (end - start) / 60, " minutes"


def ex3_process(articles_file, topics):
    # lang_voc_size = 300000
    #  The vocabulary size that was given in the assignment
    # initialize_train_data(articles_file)
    headers_train_data, articles_train_data, all_words_with_all_freqs, articles_with_their_words_freqs = create_train_data(
            articles_file)
    topics_list = get_topics(topics)
    words_clusters = split_into_clusters(articles_train_data)

    final_weights = em_process(articles_with_their_words_freqs, all_words_with_all_freqs, words_clusters, len(topics_list))
    conf_matrix = create_confusion_matrix(final_weights, articles_with_their_words_freqs,topics_list, headers_train_data)
    print conf_matrix
    accuracy = calc_accuracy()

def create_confusion_matrix(weights, articles_with_their_words_freqs,topics_list, article_topics):
    documents_in_clusters = {}

    number_of_topics = len(topics_list)
    number_of_clusters = len(weights[0].keys())

    for t in articles_with_their_words_freqs:
        max_weights = weights[t][0]
        selected_index = 0
        for i in range(0, number_of_clusters):
            if weights[t][i] > max_weights:
                max_weights = weights[t][i]
                selected_index = i
        if selected_index not in documents_in_clusters:
            documents_in_clusters[selected_index] = []
        documents_in_clusters[selected_index].append(t)


    for cluster in documents_in_clusters:
        for t in range(0, len(cluster)):
            current_topics = article_topics[t]

    conf_matrix = np.zeros((number_of_clusters, number_of_topics + 1))

    for row in range( 0, number_of_clusters):
        for col in range(0, number_of_topics):
            conf_matrix[row][col] = 0
        # Number of articles in the cluster
        conf_matrix[row][number_of_topics -1] = len(documents_in_clusters[row])
    return conf_matrix

# -------------------------------------------------- EM - ALG -------------------------------------------------- #


def calc_likelihood(m_list, z_list, k_param):
    print('likelihood')
    len_of_m_list = len(m_list)
    likelihood = 0
    for t in range(len_of_m_list):
        sum_zi_e = 0
        curr_zi_len = len(z_list[t])
        for i in range(0, curr_zi_len):
            curr_zi_m = z_list[t][i] - m_list[t]
            if curr_zi_m >= (-1.0) * k_param:
                sum_zi_e += math.exp(curr_zi_m)
        likelihood += m_list[t] + np.log(sum_zi_e)
    return likelihood


def calc_initial_alpha_and_prob(relevant_words_with_freqs, articles_with_their_freq, clusters_of_articles, number_of_clusters,
                                voc_size, lambda_val):
    alpha = [0] * number_of_clusters
    number_of_articles = len(articles_with_their_freq)
    for i in range(0, number_of_clusters):
        number_of_articles_in_i = len(clusters_of_articles[i + 1])
        alpha[i] = number_of_articles_in_i / number_of_articles

    probabilities = {}
    for word in relevant_words_with_freqs:
        probabilities[word] = {}
        for i in range(0, number_of_clusters):
            #sum_of_doc_lens = 0
            #sum_of_ntk = 0
            probabilities[word][i] = 0
            for t in clusters_of_articles[i+1]:
                current_doc = articles_with_their_freq[t]
                #sum_of_doc_lens += sum(articles_with_their_freq[t].values())
                if word in current_doc:
                    #sum_of_ntk += current_doc[word]
                    probabilities[word][i] = 1
    for word in relevant_words_with_freqs:
        curr_probs_sum = 0
        count_of_0 = 0
        for i in range(0, number_of_clusters):
            if probabilities[word][i] == 0:
                count_of_0 += 1
        for i in range(0, number_of_clusters):
            if probabilities[word][i] == 0:
                probabilities[word][i] = calc_lidstone_for_unigram(count_of_0, number_of_clusters, voc_size, lambda_val)
            else:
                probabilities[word][i] = calc_lidstone_for_unigram(number_of_clusters-count_of_0, number_of_clusters, voc_size, lambda_val)
            #probabilities[word][i] /= curr_probs_sum
    return alpha, probabilities


def calc_perplexity(lan_likelihood, voc_size ):
    return math.pow(2,(-1/voc_size * lan_likelihood))


def em_process(articles_with_their_words_freqs, all_words_with_all_freq, words_clusters, number_of_clusters):
    k_param = 10
    lambda_val = 1.1
    v_size = len(all_words_with_all_freq)
    alpha, probabilities = calc_initial_alpha_and_prob(all_words_with_all_freq, articles_with_their_words_freqs,
                                                       words_clusters, number_of_clusters, v_size,
                                                       lambda_val)
    likelihood_array = []
    perplexity_array = []
    prev_likelihood = -10000001
    curr_likelihood = -10000000
    epoch = 0
    while curr_likelihood >= prev_likelihood:
        w, z_list, m_list = e_step(all_words_with_all_freq, articles_with_their_words_freqs, alpha, probabilities,
                                   number_of_clusters, k_param)
        alpha, probabilities = m_step(w, articles_with_their_words_freqs, all_words_with_all_freq, number_of_clusters,
                                      lambda_val, v_size, words_clusters)
        prev_likelihood = curr_likelihood
        curr_likelihood = calc_likelihood(m_list, z_list, k_param)
        curr_perplexity = calc_perplexity(curr_likelihood, v_size)
        print "likelihood per curr epoch (", epoch, ") - ", curr_likelihood
        likelihood_array.append(curr_likelihood)
        perplexity_array.append(curr_perplexity)
        epoch += 1
    plot_graph(epoch, likelihood_array, "likelihood")
    plot_graph(epoch, perplexity_array, "perplexity")
    return w


def plot_graph(num_of_iterations, axis_y, label_name):
    axis_x = [i for i in range(0, num_of_iterations)]  # number of iterations
    plt.plot(axis_x, axis_y)
    plt.xlabel("iterations")
    plt.ylabel(label_name)
    plt.title("I vs L Graph")
    plt.xlim(0, num_of_iterations)
    plt.ylim(min(axis_y), max(axis_y))
    plt.legend(loc="lower left")
    plt.savefig(label_name + ".png")


def e_step(all_relevant_words, articles_with_their_words_freqs, alpha, probabilities, number_of_clusters, k_param):
    print ('e_step')
    w = {}
    z_list = []
    m_list = []
    for t, doc_with_freq in articles_with_their_words_freqs.iteritems():
        w[t] = {}
        curr_z, max_zi = calc_z_values(all_relevant_words, number_of_clusters, alpha, probabilities, doc_with_freq, k_param)
        sum_zi = 0
        for i in range(0, number_of_clusters):
            if curr_z[i] - max_zi < (-1.0) * k_param:
                w[t][i] = 0
            else:
                w[t][i] = math.exp(curr_z[i] - max_zi)
                sum_zi += w[t][i]
        for i in range(0, number_of_clusters):
            w[t][i] /= sum_zi

        z_list.append(curr_z)
        m_list.append(max_zi)
    return w, z_list, m_list


def calc_z_values(all_relevant_words, number_of_clusters, alpha, probabilities, curr_article_with_t, k_param):
    z = []
    for i in range(0, number_of_clusters):
        sum_of_freq_ln = 0
        for word in curr_article_with_t:
            sum_of_freq_ln += curr_article_with_t[word] * np.log(probabilities[word][i])
        z.append(np.log(alpha[i]) + sum_of_freq_ln)
    max_z = max(z)
    return z, max_z


def m_step(weights, articles_with_their_words_frequencies, relevant_words_with_freq, number_of_clusters, lambda_val, v_size, document_clusters):
    print('m_step')
    threshold = 0.000001
    number_of_docs = len(articles_with_their_words_frequencies)
    probabilities = {}
    denominator = []
    for i in range(0, number_of_clusters):
        denom_i = 0
        for t in articles_with_their_words_frequencies:
            len_of_t = sum(articles_with_their_words_frequencies[t].values())
            denom_i += weights[t][i] * len_of_t
        denominator.append(denom_i)
    for word in relevant_words_with_freq:
        probabilities[word] = {}
        for i in range(0, number_of_clusters):
            numerator = 0
            for t in articles_with_their_words_frequencies:
                if word in articles_with_their_words_frequencies[t] and weights[t][i] != 0:
                    numerator += weights[t][i] * articles_with_their_words_frequencies[t][word]
            probabilities[word][i] = calc_lidstone_for_unigram(numerator, denominator[i], v_size, lambda_val)
    # If alpha is smaller then a threshold we will scale it to the threshold to not get ln(alpha) = error

    alpha = [0] * number_of_clusters
    for i in range(0, number_of_clusters):
        for t in articles_with_their_words_frequencies:
            alpha[i] += weights[t][i]
        alpha[i] /= number_of_docs
    # alpha = [sum(i) / number_of_docs for i in zip(*weights)]
    for i in range(0, len(alpha)):
        if alpha[i] < threshold:
            alpha[i] = threshold
    sum_of_alpha = sum(alpha)
    # Normalize alpha for it to sum to 1
    alpha = [x / sum_of_alpha for x in alpha]
    return alpha, probabilities


def calc_weights(k_param, z_values):
    m = max(z_values)
    weights = []
    sum_e_z = 0
    for j in range(0, len(z_values)):
        if z_values[j] - m >= (-1.0) * k_param:
            sum_e_z += math.exp(z_values[j] - m)
    for i in range(0, len(z_values)):
        if z_values[i] - m < (-1.0) * k_param:
            weights.append(0)
        else:
            weights.append(math.exp(z_values[i] - m) / sum_e_z)
    return weights


# -------------------------------------------------- EM - ALG -------------------------------------------------- #

def clean_rare_words_train_data(all_relevant_words, articles_train_data):
    new_articles_train_data = {}
    for article_id, content in articles_train_data.iteritems():
        new_content = []
        for word in content:
            if word in all_relevant_words:
                new_content.append(word)
        new_articles_train_data[article_id] = new_content
    return new_articles_train_data


def create_train_data(train_file):
    headers_train_data = {}  # holds all articles' headers
    header_id = 0  # holds an id that represent a key which connect between headers dic and the articles dic
    article_id = 0
    articles_train_data = {}  # holds the articles
    all_words_with_counter = {}

    with open(train_file) as f:
        for line in f:
            splited_line = line.strip().split(' ')
            if len((splited_line[0].split('\t'))) > 1:  # note header
                headers_train_data[header_id] = splited_line
                header_id += 1
            else:  # an article
                article_content = splited_line
                articles_train_data[article_id] = article_content
                article_id += 1

                for word in article_content:
                    if word not in all_words_with_counter:
                        all_words_with_counter.setdefault(word, 1)
                    else:
                        all_words_with_counter[word] += 1

    relevant_words_with_freqs = clean_rare_words(all_words_with_counter)
    relevant_articles_train_data = clean_rare_words_train_data(relevant_words_with_freqs, articles_train_data)
    article_train_data_with_freq = get_words_freq_for_article(relevant_articles_train_data)
    return headers_train_data, relevant_articles_train_data, relevant_words_with_freqs, article_train_data_with_freq


def get_words_freq_for_article(articles_train_data):
    article_train_data_with_freq = {}
    for article_id, words in articles_train_data.iteritems():
        # articles_train_data[article_id] = new_words
        article_train_data_with_freq[article_id] = Counter(words)
    return article_train_data_with_freq


def split_into_clusters(dev_set):
    dev_split_into_clusters = {}
    for i in range(0, len(dev_set)):
        selected_cluster = (i + 1) % 9
        if selected_cluster == 0:
            selected_cluster = 9
        if selected_cluster not in dev_split_into_clusters:
            dev_split_into_clusters[selected_cluster] = []
        dev_split_into_clusters[selected_cluster].append(i)
    return dev_split_into_clusters


def get_topics(topics_file):
    topics = []
    with open(topics_file) as f:
        for line in f:
            topics.append(line.strip())
    return topics


def calc_accuracy(articles_headers, all_doc_with_classification):
    accuracy = 0
    for t, classification in all_doc_with_classification.iteritems():
        if classification in articles_headers[t]:
            accuracy += 1

    total_articles = len(articles_headers)
    return accuracy / total_articles


def calc_lidstone_for_unigram(word_occs, train_size, voc_size, lambda_value):
    # C(X)+ LAMBDA / |S| + LAMBDA*|X|
    lidstone = (word_occs + lambda_value) / (train_size + lambda_value * voc_size)
    return lidstone


# -------------------------------------------------- AUXILIARY FUNCTIONS --------------------------------------------- #

def clean_rare_words(all_words_with_counter):
    all_relevant_words = {}
    for word, occs in all_words_with_counter.iteritems():
        if occs > 3:  # ignore rare words
            all_relevant_words[word] = occs

    return all_relevant_words


def split_data_to_train_and_validate(train_fraction, all_words, article_indices):
    # This function separates the given training data to two sets - train and validation
    # according to the training_fraction we decide (for example, for lidstone - 0.9)
    train_set = {}
    s_size = len(all_words)
    validation_set = {}
    train_size = round(train_fraction * s_size)
    new_key_id = 0
    first = True
    article_ind = article_indices.values()
    for key_id, curr_word in all_words.iteritems():
        if key_id < train_size:
            if key_id in article_ind:
                train_set[new_key_id] = "begin_article"
                new_key_id += 1
            train_set[new_key_id] = curr_word
            new_key_id += 1
        else:
            # key > train_size
            if first:
                validation_set[new_key_id] = "begin_article"
                new_key_id += 1
                first = False
            if key_id in article_ind:
                validation_set[new_key_id] = "begin_article"
                new_key_id += 1
            validation_set[new_key_id] = curr_word
            new_key_id += 1
    train_unique_size = len(set(train_set.values())) - 1
    return train_set, validation_set, train_unique_size


# -------------------------------------------------- HELP FUNCTIONS -------------------------------------------------- #

if __name__ == "__main__":
    # main(sys.argv)
    main("develop.txt", "topics.txt")
