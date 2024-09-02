import matplotlib.pyplot as plt
import os

def interpolate_weights(model1, model2, baseline, alpha, device='cpu'):
    interpolated_model = baseline.to(device)
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    interpolated_state_dict = {}

    for key in state_dict1.keys():
        interpolated_state_dict[key] = alpha * state_dict1[key] + (1 - alpha) * state_dict2[key]

    interpolated_model.load_state_dict(interpolated_state_dict)
    return interpolated_model

def visualize_interpolation(alphas, error_rates, experiment):
    # error_rates *= 100

    if not os.path.exists("process_imgs"):
        os.makedirs("process_imgs")
    plt.plot(alphas, error_rates[0, :], 'r') # Eval
    plt.plot(alphas, error_rates[1, :], 'b') # Train
    plt.legend(['Eval', 'Train'])
    plt.xlabel('Interpolation')
    plt.ylabel('Error (%)')
    # plt.ylim(0, 100)
    plt.title(experiment)

    plt.grid(True)  # Enable both major and minor grid lines
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    plt.savefig(f"process_imgs/{experiment}_interpolation.png")
    plt.show()

