# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_metrics_bar_chart(metrics, values):
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=['blue', 'green', 'orange'])
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Evaluation Metrics')
    plt.ylim(0, 1)
    plt.show()
