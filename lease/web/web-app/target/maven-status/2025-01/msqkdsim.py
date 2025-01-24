import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the CSV files are located in the same directory as this script
data_protocol1 = pd.read_csv('msqkd_sim_results.csv')
data_protocol2 = pd.read_csv('msqkd_sim_results2.csv')


# Calculate detection sensitivity for each protocol
def calculate_detection_sensitivity(data):
    return data.groupby(['Noise_Level_Two', 'Noise_Level_Single'])['Eavesdropper_Detected'].mean().reset_index()


sensitivity_protocol1 = calculate_detection_sensitivity(data_protocol1)
sensitivity_protocol2 = calculate_detection_sensitivity(data_protocol2)


# Function to calculate false positive rate
def calculate_false_positive_rate(data):
    # Filter rows where Eavesdropper_Present is False
    no_eavesdropper = data[data['Eavesdropper_Present'] == False]

    # Count total cases where there's no eavesdropper
    total_no_eavesdropper = len(no_eavesdropper)

    # Count cases where Eavesdropper_Detected is False
    false_positives = len(no_eavesdropper[no_eavesdropper['Eavesdropper_Detected'] == False])

    # Calculate the false positive rate
    if total_no_eavesdropper > 0:
        false_positive_rate = (false_positives / total_no_eavesdropper) * 100
    else:
        false_positive_rate = np.nan  # Handle division by zero case if there are no such records

    return false_positive_rate


# Calculate false positive rates for both protocols
false_positive_rate1 = calculate_false_positive_rate(data_protocol1)
false_positive_rate2 = calculate_false_positive_rate(data_protocol2)

# Print the false positive rates
print(f'False Positive Rate for Our Protocol: {false_positive_rate1:.2f}%')
print(f'False Positive Rate for Lin Protocol: {false_positive_rate2:.2f}%')

# Create a bar plot for False Positive Rates
plt.figure(figsize=(8, 6))
protocols = ['Our Protocol', 'Lin Protocol']
false_positive_rates = [false_positive_rate1, false_positive_rate2]

plt.bar(protocols, false_positive_rates, color=['blue', 'orange'])
plt.xlabel('Protocol')
plt.ylabel('False Positive Rate (%)')
plt.title('False Positive Rate Comparison Between Protocols')
plt.ylim(0, max(false_positive_rates) * 1.2)  # Add some headroom above the tallest bar
plt.grid(axis='y')
plt.show()

# Plotting detection sensitivity comparison for Noise Level Single
plt.figure(figsize=(12, 6))
sns.lineplot(data=sensitivity_protocol1, x='Noise_Level_Single', y='Eavesdropper_Detected', label='Our Protocol')
sns.lineplot(data=sensitivity_protocol2, x='Noise_Level_Single', y='Eavesdropper_Detected', label='Lin Protocol')
plt.title('Eavesdropper Detection Sensitivity Comparison')
plt.xlabel('Noise Level (Single Gates)')
plt.ylabel('Detection Sensitivity')
plt.legend(title='Protocol')
plt.grid(True)
plt.show()

# Plotting detection sensitivity comparison for Noise Level Two
plt.figure(figsize=(12, 6))
sns.lineplot(data=sensitivity_protocol1, x='Noise_Level_Two', y='Eavesdropper_Detected', label='Our Protocol')
sns.lineplot(data=sensitivity_protocol2, x='Noise_Level_Two', y='Eavesdropper_Detected', label='Lin Protocol')
plt.title('Eavesdropper Detection Sensitivity Comparison')
plt.xlabel('Noise Level (Two way Gates)')
plt.ylabel('Detection Sensitivity')
plt.legend(title='Protocol')
plt.grid(True)
plt.show()


# Function to calculate false positive rate by noise levels
def calculate_false_positive_rate_by_noise_levels(data):
    # Filter rows where Eavesdropper_Present is False
    no_eavesdropper = data[data['Eavesdropper_Present'] == False]

    # Group by noise levels and calculate the false positive rate for each level
    fpr_by_noise = no_eavesdropper.groupby(['Noise_Level_Two', 'Noise_Level_Single']).apply(
        lambda df: len(df[df['Eavesdropper_Detected'] == True]) / len(df) * 100 if len(df) > 0 else np.nan
    ).reset_index(name='False_Positive_Rate')

    return fpr_by_noise


# Calculate false positive rates for both protocols
false_positive_rate1 = calculate_false_positive_rate_by_noise_levels(data_protocol1)
false_positive_rate2 = calculate_false_positive_rate_by_noise_levels(data_protocol2)

# Print the false positive rates
print('False Positive Rates for Our Protocol:')
print(false_positive_rate1)
print('\nFalse Positive Rates for Lin Protocol:')
print(false_positive_rate2)

# Merge data for plotting
fpr_combined = false_positive_rate1.rename(columns={'False_Positive_Rate': 'Our_Protocol_FPR'}).merge(
    false_positive_rate2.rename(columns={'False_Positive_Rate': 'Lin_Protocol_FPR'}),
    on=['Noise_Level_Two', 'Noise_Level_Single'],
    how='outer'
)
# Pivot tables for heatmap generation
pivot_our_protocol = fpr_combined.pivot_table(index='Noise_Level_Two', columns='Noise_Level_Single', values='Our_Protocol_FPR')
pivot_lin_protocol = fpr_combined.pivot_table(index='Noise_Level_Two', columns='Noise_Level_Single', values='Lin_Protocol_FPR')

# Define a custom colormap from red to green
cmap = sns.color_palette("RdYlGn", as_cmap=True)

# Plot heatmap for Our Protocol
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.heatmap(pivot_our_protocol, cmap=cmap, annot=True, fmt=".2f", cbar_kws={'label': 'False Positive Rate (%)'})
plt.title('False Positive Rate Heatmap (Our Protocol)')
plt.xlabel('Noise Level (Single Gates)')
plt.ylabel('Noise Level (Two-way Gates)')

# Plot heatmap for Lin Protocol
plt.subplot(1, 2, 2)
sns.heatmap(pivot_lin_protocol, cmap=cmap, annot=True, fmt=".2f", cbar_kws={'label': 'False Positive Rate (%)'})
plt.title('False Positive Rate Heatmap (Lin Protocol)')
plt.xlabel('Noise Level (Single Gates)')
plt.ylabel('Noise Level (Two-way Gates)')

plt.tight_layout()
plt.show()

