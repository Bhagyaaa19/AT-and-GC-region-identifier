'''
09. 02. 2025
Author - Bhagya Wijeratne

# Question 1
# Python implementation for identifying GC rich regions and AT rich regions - extended to handle ambiguous DNA bases
'''
# ----------------------------------------------------------------------------------------------

# Step 1: Define the problem
# Implement an HMM using Python, to classify each nucleotide of a given DNA sequence into GC-rich or AT-rich regions.

# defining the hidden states and the observations
states = ('GC-rich', 'AT-rich')

# Given DNA sequence to classify
sequence = "TTANGTTCARTGGYATTTGNACA"

# Initial state probabilities
start_probability = {'GC-rich': 0.5, 'AT-rich': 0.5}

# defining the probabilities
transition_probability = {
    'GC-rich': {'GC-rich': 0.8, 'AT-rich': 0.2},
    'AT-rich': {'GC-rich': 0.3, 'AT-rich': 0.7},
}

# Emission probabilities (likelihood of observing each nucleotide in a given state)
emission_probability = {
    'GC-rich': {'A': 0.095, 'T': 0.1, 'C': 0.4, 'G': 0.4, 'N': 0.001, 'R': 0.002, 'Y': 0.002},
    'AT-rich': {'A': 0.4, 'T': 0.4, 'C': 0.1, 'G': 0.095, 'N': 0.001, 'R': 0.002, 'Y': 0.002},
}

# Step 2: Implement the Viterbi Algorithm
def viterbi(obs_seq, states, start_prob, trans_prob, emit_prob):
    """
    Viterbi algorithm to find the most probable sequence of hidden states.
    """
    V = [{}]  # Dynamic programming table
    path = {}  # Stores best path

    # Initialize base case (t=0)
    for state in states:
        V[0][state] = start_prob[state] * emit_prob[state].get(obs_seq[0], 1e-10)
        path[state] = [state]

    # Run Viterbi for t > 0
    for t in range(1, len(obs_seq)):
        V.append({})
        new_path = {}

        for curr_state in states:
            (prob, prev_state) = max(
                (V[t - 1][prev_state] * trans_prob[prev_state][curr_state] * emit_prob[curr_state].get(obs_seq[t], 1e-10), prev_state)
                for prev_state in states
            )
            V[t][curr_state] = prob
            new_path[curr_state] = path[prev_state] + [curr_state]

        path = new_path  # Update path

    (final_prob, final_state) = max((V[len(obs_seq) - 1][state], state) for state in states)
    return path[final_state]

# Step 3: Run the Viterbi algorithm on the DNA sequence
predicted_states = viterbi(sequence, states, start_probability, transition_probability, emission_probability)

# Step 4: Identify and separate GC-rich and AT-rich regions
def identify_regions(sequence, predicted_states):
    regions = []
    current_state = predicted_states[0]
    current_segment = ""

    for i, state in enumerate(predicted_states):
        if state == current_state:
            current_segment += sequence[i]
        else:
            regions.append((current_state, current_segment))
            current_state = state
            current_segment = sequence[i]

    regions.append((current_state, current_segment))  # Append last segment
    return regions

regions = identify_regions(sequence, predicted_states)

# Output the result
print("Given DNA sequence: ", sequence)
print("Predicted Regions:")
for region in regions:
    print(f"{region[0]}: {region[1]}")
