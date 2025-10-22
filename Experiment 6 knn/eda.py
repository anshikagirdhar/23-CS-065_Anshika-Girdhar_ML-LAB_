import matplotlib.pyplot as plt
import numpy as np
from data import load_iris_data

def create_feature_scatter_plots():
    X, y = load_iris_data()
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            
            for species in np.unique(y):
                mask = y == species
                ax.scatter(X[mask, j], X[mask, i], 
                          label=species, 
                          alpha=0.6)
            
            ax.set_xlabel(feature_names[j])
            ax.set_ylabel(feature_names[i])
            ax.set_title(f'{feature_names[i]} vs {feature_names[j]}')
            
            if i == 0 and j == 0:
                ax.legend()
    
    plt.tight_layout()
    plt.show()

create_feature_scatter_plots()