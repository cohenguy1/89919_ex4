from __future__ import division
import numpy as np
import sys
import math
from collections import Counter
import time
import matplotlib.pyplot as plt


def main(argv):
    ex3_process(argv[1], argv[2])


def assign_classifications_to_docs(clusters_with_topics,documents_in_clusters):
    docs_with_assignments = []
    for index_of_cluster in documents_in_clusters:
        for t in documents_in_clusters[index_of_cluster]:
            classification = clusters_with_topics[index_of_cluster]
            docs_with_assignments.append((t, classification))
    return docs_with_assignments


def ex3_process(articles_file, topics):
    headers_train_data, articles_train_data, all_words_with_all_freqs, articles_with_their_words_freqs = create_train_data(
            articles_file)
    topics_list = get_topics(topics)
    words_clusters = split_into_clusters(articles_train_data)

    # Run the em algorithm to find the best wti for all docs according to the train data with the specific parameters
    # we gave
    final_weights = em_process(articles_with_their_words_freqs, all_words_with_all_freqs, words_clusters, len(topics_list))
    #Create the conf matrix from the best weights
    conf_matrix, clusters_with_topics, documents_in_clusters = create_confusion_matrix(final_weights, articles_with_their_words_freqs,topics_list, headers_train_data)
    print conf_matrix
    docs_with_classification = assign_classifications_to_docs(clusters_with_topics,documents_in_clusters)
    print "\n"
    accuracy = calc_accuracy(headers_train_data, docs_with_classification)
    print "accuracy - ", accuracy


def create_confusion_matrix(weights, articles_with_their_words_freqs,topics_list, article_topics):
    documents_in_clusters = {}

    number_of_topics = len(topics_list)
    number_of_clusters = len(weights[0].keys())

    # Set each cluster with it's corresponding cluster by wti
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

    # Create the confusion matrix
    conf_matrix = np.zeros((number_of_clusters, number_of_topics + 1))
    for row in range(0, number_of_clusters):
        for col in range(0, number_of_topics):
            current_topic = topics_list[col]
            for t in documents_in_clusters[row]:
                if current_topic in article_topics[t]:
                    conf_matrix[row][col] += 1
        # Number of articles in the cluster
        conf_matrix[row][number_of_topics] = len(documents_in_clusters[row])

    clusters_with_topics = {}
    for row in range(0, number_of_clusters):
        dominant_topic = 0
        dominant_topic_val = 0
        for col in range(0, number_of_topics):
            if conf_matrix[row][col] > dominant_topic_val:
                dominant_topic = topics_list[col]
                dominant_topic_val = conf_matrix[row][col]
        clusters_with_topics[row] = dominant_topic

    return conf_matrix, clusters_with_topics, documents_in_clusters

# -------------------------------------------------- EM - ALG -------------------------------------------------- #


def calc_likelihood(m_list, z_list, k_param):
    #print('likelihood')
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

    weights = {}
    # Initialize the weights by the initial clusters - if the doc in the cluster - set the weight to 1, else - set to 0
    for i, doc_list in clusters_of_articles.iteritems():
        for t in doc_list:
            weights[t] = {}
            weights[t][i-1] = 1
            for m in range(0, number_of_clusters):
                if m not in weights[t]:
                    weights[t][m] = 0

    # Initialize the alpha and probs like in m step
    alpha, probabilities = m_step(weights, articles_with_their_freq, relevant_words_with_freqs, number_of_clusters, lambda_val, voc_size)
    return alpha, probabilities


# Calculate perplexity by the given formula of th exercise
def calc_perplexity(lan_likelihood, number_of_words):
    return math.pow(2, (-1 / number_of_words * lan_likelihood))


def em_process(articles_with_their_words_freqs, all_words_with_all_freq, words_clusters, number_of_clusters):
    k_param = 10
    lambda_val = 1.1
    em_threshold = 10
    v_size = len(all_words_with_all_freq)
    # First we will initialize the Pik and Alpha_i for the model
    alpha, probabilities = calc_initial_alpha_and_prob(all_words_with_all_freq, articles_with_their_words_freqs,
                                                       words_clusters, number_of_clusters, v_size,
                                                       lambda_val)

    likelihood_array = []
    perplexity_array = []
    # Initial value for the algorithm to continue running
    prev_likelihood = -10000101
    curr_likelihood = -10000000
    epoch = 0
    number_of_words = sum(all_words_with_all_freq.values())
    # The em will continue running until the current calculated likelihood is smaller from the previous calculated
    # likelihood
    while curr_likelihood - prev_likelihood > em_threshold:
        # In the e-step the algorithm calculates the weights of each document to be in a cluster
        # And returns them and the list of z and m (for the likelihood)
        w, z_list, m_list = e_step(all_words_with_all_freq, articles_with_their_words_freqs, alpha, probabilities,
                                   number_of_clusters, k_param)
        # In the m-step the algorithm calculates the alphas and probs according to the givem weight values
        alpha, probabilities = m_step(w, articles_with_their_words_freqs, all_words_with_all_freq, number_of_clusters,
                                      lambda_val, v_size)
        prev_likelihood = curr_likelihood
        # Calc the lan likelihood of the model
        curr_likelihood = calc_likelihood(m_list, z_list, k_param)
        # Calc the model's perplexity
        curr_perplexity = calc_perplexity(curr_likelihood, number_of_words)

        likelihood_array.append(curr_likelihood)
        perplexity_array.append(curr_perplexity)
        epoch += 1

    # Create the graphs of the likelihood and perplexity per epoch
    plot_graph(epoch, likelihood_array, "likelihood")
    plot_graph(epoch, perplexity_array, "perplexity")

    # Return the final weights
    return w


def plot_graph(num_of_iterations, axis_y, label_name):
    axis_x = [i for i in range(0, num_of_iterations)]  # number of iterations
    plt.plot(axis_x, axis_y, label=label_name)
    plt.xlabel("iterations")
    plt.ylabel(label_name)
    plt.title("I vs L Graph")
    plt.xlim(0, num_of_iterations)
    plt.ylim(min(axis_y), max(axis_y))
    plt.legend(loc="lower left")
    plt.savefig(label_name + ".png")


def e_step(all_relevant_words, articles_with_their_words_freqs, alpha, probabilities, number_of_clusters, k_param):
    w = {}
    z_list = []
    m_list = []
    # For every document in our training set we want to calculate it's possibility to be in the clusters
    for t, doc_with_freq in articles_with_their_words_freqs.iteritems():
        w[t] = {}
        # Calculate the z array (for every cluster there is it's own value of z) for the current doc
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


# Calculate the z array for a current doc by the equation
def calc_z_values(all_relevant_words, number_of_clusters, alpha, probabilities, curr_article_with_t, k_param):
    z = []
    for i in range(0, number_of_clusters):
        sum_of_freq_ln = 0
        for word in curr_article_with_t:
            sum_of_freq_ln += curr_article_with_t[word] * np.log(probabilities[word][i])
        z.append(np.log(alpha[i]) + sum_of_freq_ln)
    max_z = max(z)
    return z, max_z


def m_step(weights, articles_with_their_words_frequencies, relevant_words_with_freq, number_of_clusters, lambda_val, v_size):
    #print('m_step')
    threshold = 0.000001
    number_of_docs = len(articles_with_their_words_frequencies)
    probabilities = {}
    denominator = []
    # For every cluster we want to calculate the new probs for words to be in the clusters
    # The calculation is by the wti (the probability of the doc to be in the cluster)
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


# -------------------------------------------------- EM - ALG -------------------------------------------------- #

# As suggested in the assignment we removed words that appeared in the training data less or equal to 3 times
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
                headers_train_data[header_id] = splited_line[0].replace("<", "").replace(">", "").split("\t")
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

    # We want to use only words that appeared more then 3 times
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


# The splitting is done by a simple method - we iterate through all the articles and set the article to the cluster by
# the doc index, the cluster number is doc_index%9 (9 - number of the clusters)
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


# Calculate the accuracy of the model
def calc_accuracy(articles_headers, all_doc_with_classification):
    accuracy = 0
    for doc in all_doc_with_classification:
        if doc[1] in articles_headers[doc[0]]:
            accuracy += 1

    total_articles = len(articles_headers)
    return accuracy / total_articles


# Will be helpful in smoothing
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
     main(sys.argv)
    #main("develop.txt", "topics.txt")
