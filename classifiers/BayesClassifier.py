from scipy.stats import multivariate_normal
import numpy as np
class BivariateBayesClassifier():
    def __init__(self):
        self.classes = []
        self.class_parameters = {}

    def fit(self,df,covarianceNature,diagonals,uniformCovarianceType='classwise'):
        """
            Fit the classifier to the data.

            Args:
                df (DataFrame): The input dataframe with features and class labels.
                covarianceNature (str): The nature of covariance matrices.
                    - 'same': All classes share the same covariance matrix.
                    - 'different': Each class has a different covariance matrix.
                diagonals (str): The type of covariance matrix diagonalization.
                    - 'diagonal': Diagonal covariance matrix.
                    - 'full': Full covariance matrix.
                uniformCovarianceType (str, optional): The type of covariance matrix uniformity.
                    - 'classwise': Covariance matrices obtained by taking average of convariance matriced of all the classes.
                    - 'entire': Covariance matrices obtained by taking average of entire training data of all the classes combined.

            Returns:
                None
        """
         
        self.classes = df['className'].unique()
        self.class_parameters = {}

        # covarianceNature = ['same','different']
        # diagonals = ['full','diagonal']
        class_covariances = {}
        for c in self.classes:
            class_df = df[df['className']==c]
            class_covariances[c] = self.covariance(class_df)

        for c in self.classes:
            class_df = df[df['className'] == c]
            class_mean = class_df[['col1', 'col2']].mean().values

            class_covariance = np.zeros((2,2))
            if covarianceNature == 'same':
                if uniformCovarianceType == 'classwise':
                    class_covariance =  np.mean(list(class_covariances.values()),axis=0)
                elif uniformCovarianceType == 'entire':
                    class_covariance = self.covariance(df)
            elif covarianceNature == 'different':
                class_covariance = class_covariances[c]
            
            if diagonals == 'diagonal':
                class_covariance = self.diagonalize(class_covariance)

            
            self.class_parameters[c] = {
                'mean': class_mean,
                'covariance': class_covariance,
                'prior': len(class_df) / len(df)
            }
        
    
    def covariance(self,df):
        return df[['col1','col2']].cov().values
    
    def diagonalize(self,cov_matrix):
        cov_matrix = np.array(cov_matrix)
        return cov_matrix * np.identity(cov_matrix.shape[0])
    
    def predict(self,x):
        predicted_classes = []

        for dataPoint in x:
            posteriors = []
            for c in self.classes:
                class_params = self.class_parameters[c]
                prior = class_params['prior']
                mean = class_params['mean']
                covariance = class_params['covariance']
                
                mvn = multivariate_normal(mean=mean, cov=covariance)
                posterior = prior * mvn.pdf(dataPoint)
                posteriors.append(posterior)

            predicted_class = self.classes[np.argmax(posteriors)]
            predicted_classes.append(predicted_class)
        return np.array(predicted_classes)
    
