import numpy as np
class ReferenceTemplateClassifier_Mahalodian():
    def __init__(self):
        self.mean = {}
        self.covariance = {}
        self.className = []
    
    def fit(self, df):
        groups = df.groupby('className')
        self.className = df.className.unique()
        for name, group in groups:
            self.mean[name] = group[['col1','col2']].mean().to_numpy()
            self.covariance[name] = group[['col1','col2']].cov().to_numpy()
        
    
    def predict(self, data_array):
        y_pred = []
        for row in data_array:
            min_distance = float('inf')
            assigned_class = None

            for name in self.className:
                distance = np.sqrt(
                    np.dot(np.dot((row - self.mean[name]).T, np.linalg.inv(self.covariance[name])), (row - self.mean[name]))
                )
                if distance < min_distance:
                    min_distance = distance
                    assigned_class = name
            y_pred.append(assigned_class)
        return np.array(y_pred)
    