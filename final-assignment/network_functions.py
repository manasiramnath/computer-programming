from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

def main():
    pass

def get_data(fname):
    """Open Wikipedia revert data in file fname and return
    a dictonary where keys are articles and values is a nested list of edits."""
    
    with open(fname, 'r') as f:
        f.readline() 
        wiki_dict = {}
        for edit in f.readlines():
            title, dt, rev, version, user = edit.strip().split('\t')
            if title not in wiki_dict:
                wiki_dict[title] = []
            wiki_dict[title].append([datetime.strptime(dt, "%Y-%m-%d %H:%M:%S"), 
                                int(rev), int(version), user])
    return wiki_dict

def save_data(wiki_dict, filename):
    """Pickle data and save to file."""

    with open(filename, 'wb') as file:
        pickle.dump(wiki_dict, file)
    return None

def load_data(filename):
    """Unpickle data and load from file."""

    with open(filename, 'rb') as file:
        wiki_dict = pickle.load(file)
    return wiki_dict

def get_sorted_edits(wiki_dict):
    """Given a dictionary data, extracts a list of edits sorted by date and 
    returns a dictionary where key is user and value is a nested list of edit times in chronological order. """

    # Extract user and dates as a list sorted by date
    sorted_edits = [[edit[0], edit[3]] for edits in wiki_dict.values() for edit in edits]
    sorted_edits.sort(key=lambda x: x[0])

    sorted_user_edits = {}
    for time, user in sorted_edits:
        sorted_user_edits.setdefault(user, []).append([time])

    return sorted_user_edits

def create_network(wiki_dict):
    """Creates a network as an edge list of dictionaries, representing reverts in a Wikipedia dataset.

    The function iterates through keys and values in a dictionary 'wiki_dict' where keys are articles, and values are lists of edits under that article.

    Valid reverts are identified by checking two conditions:
    1. Edits but no change: Removes edits where a user reverts to the same version another user made immediately after them.
    2. Self-reverts: Removes edits where the same editor reverts to an earlier version of an article and makes the revision immediately after that version.

    If reverts pass both conditions, the function appends reverter, reverted, and time of revert to a network list.

    Args:
        wiki_dict (dict): A dictionary containing articles as keys and lists of edits under each article as values.

    Returns:
        network_before_seniority: An edge-list of dictionaries with keys 'reverter', 'reverted', and 'time', representing revert events.
    """

    network_before_seniority = []
    for article, edits in wiki_dict.items():
        ind = 0
        while ind < len(edits):
            edit = edits[ind]
            if edit[1] == 1:
                dt, version, reverter = edit[0], edit[2], edit[3]
                ## Removes edits
                # Case 1: Edits but no change
                if ind + 1 < len(edits) and edits[ind + 1][2] == version and edits[ind + 1][3] != reverter:
                    pass
                # Case 2: Self-reverts
                else:
                    for j in range(ind + 1, len(edits)):
                        if j < len(edits) and edits[j][2] == version and edits[j-1][3] == reverter:
                            break
                        # Append name of reverter, user of j-1 who is the reverted and time of revert
                        elif edits[j][2] == version:
                            network_before_seniority.append({'reverter': reverter, 'reverted': edits[j - 1][3], 'time': dt})
                            break 
                    
            ind += 1    
    return network_before_seniority

def seniority(sorted_user_edits, network_before_seniority):
    """Given a sorted list of edits and a network list of dictionaries, 
    calculates the seniority of editor i (reverter) and editor j (reverted) at time of revert.

    Args:
    sorted_user_edits (dict): A dictionary where key is user and value is a nested list of edit times in chronological order.
    network_before_seniority (lst): A network list of dictionaries with keys 'reverter', 'reverted', and 'time'.

    Returns the network list of dictionaries with 'seniority_i', and 'seniority_j' keys appended to each dictionary.
    """

    # Iterate through network list of dictionaries
    for revert in network_before_seniority:
        editor_i = revert['reverter']
        editor_j = revert['reverted']
        revert_time = revert['time']

        # Get edits of reverter and reverted
        reverter_edits = sorted_user_edits[editor_i]
        reverted_edits = sorted_user_edits[editor_j]

        edits_before_revert, edits_before_reverted = 0, 0
        i, j = 0, 0
        
        # Count number of edits before revert and reverted
        while i < len(reverter_edits) and reverter_edits[i][0] < revert_time:
            edits_before_revert += 1
            i += 1
    
        while j < len(reverted_edits) and reverted_edits[j][0] < revert_time:
            edits_before_reverted += 1
            j += 1
        
        # Calculate seniority 
        # If no edits before revert or reverted, seniority is 0 and 1 is added to avoid log(0)
        if edits_before_revert == 0 or edits_before_reverted == 0:
            seniority_i = math.log10(edits_before_revert + 1)
            seniority_j = math.log10(edits_before_reverted + 1)
        else:
            seniority_i = math.log10(edits_before_revert)
            seniority_j = math.log10(edits_before_reverted)

        # Append seniority to network list of dictionaries
        revert['seniority_i'] = seniority_i
        revert['seniority_j'] = seniority_j
            
    return network_before_seniority

def get_nodes_edges(network):
    """Given a network list of dictionaries, print first 5 edges, number of nodes and number of edges."""

    print("First 5 points:")
    for edge in network[:5]:
        print(edge)
    nodes = set()
    for edge in network:
        nodes.add(edge['reverter'])
        nodes.add(edge['reverted'])

    print("Number of nodes:", len(nodes))
    print("Number of edges:", len(network))
    return None

def get_abba(network):
    """Identifies AB-BA sequences in a network of revert events.

    Given a list of revert events in the form of dictionaries, this function looks for the two-event sequence where after
    editor A reverts editor B, B reverts A back within a 24-hour window.
    Returns a list of dictionaries containing information about the edges.

    Args:
        network (list): An edge-list of dictionaries representing reverts. Each dictionary is an edge that should contain the following keys:
            - 'reverter': The name of the editor who reverted another editor.
            - 'reverted': The name of the editor who was reverted.
            - 'time': The timestamp of the revert event as a datetime object.

    Returns:
        abba_sequences (list): A list of dictionaries containing the following keys about identified AB-BA sequences:
            - 'reverter': Editor who reverted another editor (A).
            - 'reverted': Editor who was reverted (B).
            - 'reverter1': Editor who reverted editor back (B).
            - 'reverted1': Editor who was reverted back (A).
            - 'ab_time': The time A reverts B as a datetime object.
            - 'ba_time': The time B reverts A back as a datetime object.
    """

    # Sort the network by time
    sorted_network = sorted(network, key=lambda x: x['time'])

    abba_sequences = []

    for i, edge in enumerate(sorted_network):
        a = edge['reverter']
        b = edge['reverted']
        dt = edge['time']

        # Find the first BA edge within 24 hours of AB edge
        for edge2 in sorted_network[i+1:]:
            if edge2['reverter'] == b and edge2['reverted'] == a and timedelta(days=0) < (edge2['time'] - dt) < timedelta(days=1):
                abba_sequences.append({'reverter': a, 'reverted': b, 'reverter1': b, 'reverted1': a, 'ab_time': dt, 'ba_time': edge2['time']})
                break

    print(f'Number of AB-BA sequences: {len(abba_sequences)}')
    return abba_sequences



def compare_seniority(network, abba_sequences):
    """Compares absolute difference in seniority between editors involved in AB-BA event sequences with 
    the absolute difference in seniority between editors involved in non-AB-BA event sequences.

    Args:
        network (list): An edge-list of dictionaries representing reverts. Each dictionary is an edge that should contain the following keys:
            - 'reverter': The name of the editor who reverted another editor.
            - 'reverted': The name of the editor who was reverted.
            - 'time': The timestamp of the revert event as a datetime object.
            - 'seniority_i': The seniority of the editor who reverted another editor.
            - 'seniority_j': The seniority of the editor who was reverted.
        abba_sequences (list): A list of dictionaries containing the following keys about identified AB-BA sequences:
            - 'reverter': Editor who reverted another editor (A).
            - 'reverted': Editor who was reverted (B).
            - 'reverter1': Editor who reverted editor back (B).
            - 'reverted1': Editor who was reverted back (A).
            - 'ab_time': The time A reverts B as a datetime object.
            - 'ba_time': The time B reverts A back as a datetime object.
    
    Returns:
        abba_seniority_diff (list): A list of absolute difference in seniority between editors involved in AB-BA event sequences.
        non_abba_seniority_diff (list): A list of absolute difference in seniority between editors not involved in AB-BA event sequences.
    """

    abba_edges = []
    
    for edge in network:
        for abba in abba_sequences:
            if (
                (edge['reverter'] == abba['reverter'] and edge['reverted'] == abba['reverted'] and edge['time'] == abba['ab_time']) or
                (edge['reverter'] == abba['reverter1'] and edge['reverted'] == abba['reverted1'] and edge['time'] == abba['ba_time'])
            ):
                abba_edges.append(edge)
                break

    abba_seniority_diff = [abs(edge['seniority_i'] - edge['seniority_j']) for edge in abba_edges]
    non_abba_seniority_diff = [abs(edge['seniority_i'] - edge['seniority_j']) for edge in network if edge not in abba_edges]

    return abba_seniority_diff, non_abba_seniority_diff


def plot_seniority_stats(abba_seniority_diff, non_abba_seniority_diff):
    """Plots histograms and prints mean absolute seniority differences for editors involved in AB-BA and non-AB-BA event sequences.

    Args:
        abba_seniority_diff (list): Absolute differences in seniority between editors in AB-BA event sequences.
        non_abba_seniority_diff (list): Absolute differences in seniority between editors in non-AB-BA event sequences.

    Returns:
        None
    """
    # Plot histogram of |sA - sB| for reverts that are part of AB–BA motifs on top of a histogram of |si - sj| for all other reverts
    plt.hist(non_abba_seniority_diff, bins=20, alpha=0.5, label='Non AB-BA', density=True)
    plt.hist(abba_seniority_diff, bins=20, alpha=0.5, label='AB-BA', density=True)
    plt.legend(loc='upper left')
    plt.xlabel('Absolute difference in seniority')
    plt.ylabel('Density')
    plt.title('Histogram of absolute difference in seniority between editors involved in AB-BA and non-AB-BA event sequences')
    plt.show()

    # Print the mean |sA - sB| and the mean |si - sj|
    print(f'Mean absolute seniority difference for editors involved in AB-BA event sequences: {np.mean(abba_seniority_diff)}')
    print(f'Mean absolute seniority difference for editors involved in non-AB-BA event sequences: {np.mean(non_abba_seniority_diff)}')
    return None
