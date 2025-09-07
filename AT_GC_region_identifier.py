'''
09. 02. 2025
Author - Bhagya Wijeratne

# Question 1
# Python implementation for identifying GC-rich and AT-rich regions separately
'''
# ----------------------------------------------------------------------------------------------

# Step 1: Define the problem
# Implement an HMM using Python to classify each nucleotide of a given DNA sequence into GC-rich or AT-rich regions.

# Defining the hidden states and the observations
states = ('GC-rich', 'AT-rich')

# Given DNA sequence to classify
sequence = "TTAAGTTCACTGGTATTTGAACA"

# Initial state probabilities
start_probability = {'GC-rich': 0.5, 'AT-rich': 0.5}

# Transition probabilities
transition_probability = {
    'GC-rich': {'GC-rich': 0.8, 'AT-rich': 0.2},
    'AT-rich': {'GC-rich': 0.3, 'AT-rich': 0.7},
}

# Emission probabilities (likelihood of observing each nucleotide in a given state)
emission_probability = {
    'GC-rich': {'A': 0.1, 'T': 0.1, 'C': 0.4, 'G': 0.4},
    'AT-rich': {'A': 0.35, 'T': 0.35, 'C': 0.15, 'G': 0.15},
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
        V[0][state] = start_prob[state] * emit_prob[state][obs_seq[0]]
        path[state] = [state]

    # Run Viterbi for t > 0
    for t in range(1, len(obs_seq)):
        V.append({})
        new_path = {}

        for curr_state in states:
            # Choose the best previous state that leads to this one
            (prob, prev_state) = max(
                (V[t - 1][prev_state] * trans_prob[prev_state][curr_state] * emit_prob[curr_state][obs_seq[t]],
                 prev_state)
                for prev_state in states
            )

            V[t][curr_state] = prob
            new_path[curr_state] = path[prev_state] + [curr_state]

        path = new_path  # Update path

    # Find the final most probable state
    (final_prob, final_state) = max((V[len(obs_seq) - 1][state], state) for state in states)

    return path[final_state]  # Return the most probable state sequence

# Step 3: Run the Viterbi algorithm on the DNA sequence
predicted_states = viterbi(sequence, states, start_probability, transition_probability, emission_probability)

# Step 4: Identify and separate GC-rich and AT-rich regions
def identify_regions(sequence, predicted_states):
    regions = []
    current_state = predicted_states[0]
    start = 0

    for i in range(1, len(sequence)):
        if predicted_states[i] != current_state:
            regions.append((current_state, sequence[start:i]))
            current_state = predicted_states[i]
            start = i

    regions.append((current_state, sequence[start:]))  # Add the last segment
    return regions

# Get the regions
regions = identify_regions(sequence, predicted_states)

# Output the result
print("Given DNA sequence: ", sequence)
for region in regions:
    print(f"{region[0]} region: {region[1]}")
