import matplotlib.pyplot as plt
# Read discriminator loss values
def plot_SN_GAN_Loss():
    with open("avg_loss_D.txt", 'r') as f:
        score_D = [float(line.rstrip('\n')) for line in f]

    # Read generator loss values
    with open("avg_loss_G.txt", 'r') as f:
        score_G = [float(line.rstrip('\n')) for line in f]

    # Generate lists of epoch numbers based on the length of the scores
    epochs = list(range(1, len(score_D) + 1))

    # Plotting the first 300 epochs
    plt.figure(figsize=(20, 12))  # Set the figure size for two stacked plots

    plt.subplot(2, 1, 1)  # First subplot
    plt.plot(epochs, score_D, label="Discriminator Loss", color='blue', markerfacecolor='red', linestyle='-', marker='o', markersize=4)
    plt.plot(epochs, score_G, label="Generator Loss", color='orange', markerfacecolor='black', linestyle='-', marker='o', markersize=4)
    plt.xlabel('Epoch (First 300)')
    plt.ylabel('Average Loss')
    plt.title('Discriminator and Generator Loss vs Epochs (First 300)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

plot_SN_GAN_Loss()