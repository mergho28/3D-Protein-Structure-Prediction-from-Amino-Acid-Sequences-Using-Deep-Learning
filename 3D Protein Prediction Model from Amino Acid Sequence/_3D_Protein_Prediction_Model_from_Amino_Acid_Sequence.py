import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Map one-letter amino acid codes to full names
amino_acid_full_names = {
    'A': 'Alanine', 'C': 'Cysteine', 'D': 'Aspartic Acid', 'E': 'Glutamic Acid',
    'F': 'Phenylalanine', 'G': 'Glycine', 'H': 'Histidine', 'I': 'Isoleucine',
    'K': 'Lysine', 'L': 'Leucine', 'M': 'Methionine', 'N': 'Asparagine',
    'P': 'Proline', 'Q': 'Glutamine', 'R': 'Arginine', 'S': 'Serine',
    'T': 'Threonine', 'V': 'Valine', 'W': 'Tryptophan', 'Y': 'Tyrosine'
}

# Generate a simple amino acid encoder (convert each amino acid to a numeric vector)
def encode_sequence(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    encoding = np.zeros((len(sequence), len(amino_acids)))
    for i, aa in enumerate(sequence):
        encoding[i, amino_acids.index(aa)] = 1
    return encoding

# Create a simple neural network model to predict 3D coordinates
def build_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='linear')  # 3D coordinates as output
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Visualize the predicted 3D structure with full amino acid names in the legend
def visualize_structure_with_legend(sequence, coords, title="Predicted 3D Structure"):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # All amino acids
    fig = plt.figure(facecolor='black')  # Set figure background to black
    ax = fig.add_subplot(111, projection='3d', facecolor='black')  # Set plot background to black
    coords = np.array(coords)
    
    # Dictionary to assign a unique color to each amino acid
    color_map = plt.cm.get_cmap('tab20', len(amino_acids))  # Color map for amino acids
    
    # Plot each amino acid as a sphere with its own color
    for i, aa in enumerate(sequence):
        color = color_map(amino_acids.index(aa) / len(amino_acids))  # Unique color for each amino acid
        ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2], s=100, color=color, label=aa)
        ax.text(coords[i, 0], coords[i, 1], coords[i, 2], aa, color='white', fontsize=10)

    # Add a thinner line connecting all amino acids in the sequence to represent the structure
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color='white', linestyle='-', linewidth=0.5)  # Thinner line

    # Add legend with full amino acid names instead of one-letter codes
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map(i / len(amino_acids)),
                          markersize=10, label=amino_acid_full_names[aa]) for i, aa in enumerate(amino_acids)]
    
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10, title="Amino Acids")

    # Set axis colors and grid to white
    ax.xaxis._axinfo['grid'].update(color = 'white')
    ax.yaxis._axinfo['grid'].update(color = 'white')
    ax.zaxis._axinfo['grid'].update(color = 'white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    ax.set_title(title, color='white')  # Set title color to white
    plt.show()

# Function to simulate data and predictions
def train_and_predict(sequence):
    # Encode the sequence
    encoded_seq = encode_sequence(sequence)
    
    # Generate some synthetic target coordinates (since we don't have real data)
    np.random.seed(0)
    true_coords = np.random.rand(len(sequence), 3) * 10
    
    # Build and train the model
    model = build_model(encoded_seq.shape[1])
    model.fit(encoded_seq, true_coords, epochs=500, verbose=0)
    
    # Predict the 3D structure
    predicted_coords = model.predict(encoded_seq)
    
    # Print and visualize the predicted 3D structure
    print("Predicted 3D Coordinates:")
    print(predicted_coords)
    
    visualize_structure_with_legend(sequence, predicted_coords)

# Example amino acid sequence
amino_acid_sequence = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"  # input amino acid sequence

# Train the model and predict 3D structure
train_and_predict(amino_acid_sequence)






