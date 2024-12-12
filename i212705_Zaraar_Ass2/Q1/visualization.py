import matplotlib.pyplot as plt

def plot_VAE_Loss():
    with open("avg_loss.txt", 'r') as f:
        score = [int(float(line.rstrip('\n')))//100 for line in f]

    avg_loss = score[99:]

    # Generate a list of epoch numbers based on the length of avg_loss
    epochs = list(range(100, len(score) + 1))

    # Plotting the avg_loss values over epochs
    plt.figure(figsize=(20, 6))  # Increased width to 15
    plt.plot(epochs, avg_loss, label="Average Loss", markerfacecolor='red',color='black', marker='o', linestyle='-', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('VAE Average Loss vs Epochs')
    # plt.grid(True)
    plt.legend()
    plt.show()



# Read discriminator loss values
def plot_GAN_Loss():
    with open("avg_loss_D.txt", 'r') as f:
        score_D = [float(line.rstrip('\n')) for line in f]

    # Read generator loss values
    with open("avg_loss_G.txt", 'r') as f:
        score_G = [float(line.rstrip('\n')) for line in f]

    # Generate lists of epoch numbers based on the length of the scores
    epochs = list(range(1, len(score_D) + 1))

    # Splitting data into two halves for plotting
    epochs_first_half = epochs[:300]
    score_D_first_half = score_D[:300]
    score_G_first_half = score_G[:300]

    epochs_second_half = epochs[300:]
    score_D_second_half = score_D[300:]
    score_G_second_half = score_G[300:]

    # Plotting the first 300 epochs
    plt.figure(figsize=(20, 12))  # Set the figure size for two stacked plots

    plt.subplot(2, 1, 1)  # First subplot
    plt.plot(epochs_first_half, score_D_first_half, label="Discriminator Loss", color='blue', markerfacecolor='red', linestyle='-', marker='o', markersize=4)
    plt.plot(epochs_first_half, score_G_first_half, label="Generator Loss", color='orange', markerfacecolor='black', linestyle='-', marker='o', markersize=4)
    plt.xlabel('Epoch (First 300)')
    plt.ylabel('Average Loss')
    plt.title('Discriminator and Generator Loss vs Epochs (First 300)')
    plt.grid(True)
    plt.legend()

    # Plotting the second 300 epochs
    plt.subplot(2, 1, 2)  # Second subplot
    plt.plot(epochs_second_half, score_D_second_half, label="Discriminator Loss", color='blue', markerfacecolor='red', linestyle='-', marker='o', markersize=4)
    plt.plot(epochs_second_half, score_G_second_half, label="Generator Loss", color='orange', markerfacecolor='black', linestyle='-', marker='o', markersize=4)
    plt.xlabel('Epoch (Last 300)')
    plt.ylabel('Average Loss')
    plt.title('Discriminator and Generator Loss vs Epochs (Last 300)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

plot_VAE_Loss()
plot_GAN_Loss()