import numpy as np
from sklearn.metrics import r2_score
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import det
from tqdm import tqdm
class InfluentialPointsDetector:
    def __init__(self, threshold=0.01):
        """
        Initialize the detector with a threshold for R² score change.
        Points causing an R² decrease above this threshold are considered influential.
        """
        self.threshold = threshold

    def detect(self,X, true_labels, predictions):
        """
        Detect points that significantly affect the R² score.
        :param true_labels: Array of true labels.
        :param predictions: Array of predicted values.
        :return: Indices of influential points.
        """
        # Calculate the overall R² score
        overall_r2 = r2_score(true_labels, predictions)

        influential_points = []
        for i in range(len(true_labels)):
            # Exclude the i-th point
            temp_true_labels = np.delete(true_labels, i, axis=0)
            temp_predictions = np.delete(predictions, i, axis=0)

            # Calculate the R² score without the i-th point
            temp_r2 = r2_score(temp_true_labels, temp_predictions)

            # Check if the exclusion of the point significantly changes the R² score
            if overall_r2 - temp_r2 > self.threshold:
                influential_points.append(i)
        if len(influential_points)>0:
            influential_points = np.array(influential_points)
            print(f"Num of influential points/ len data: {len(influential_points)}/{len(true_labels)} ")
            with open('influential_points.npy','wb') as f:
                np.save(f,X[influential_points])
            with open('influential_points_labels.npy','wb') as f:
                np.save(f,true_labels[influential_points])
            print("Saved the data and labels")
            
        return influential_points
    
    def detect_two_distributions(self, X1 , X2):
        """
        Detect the points in X1 having similar distribution with X2
        Args:
            X1: X1 distribution
            X2: X2 distribution
            return: indexes of points in X1 make it different from X2
        """
        ## using bhattacharyya distance
        def bhattacharyya_distance(mu1, sigma1, mu2, sigma2):
            # mu1, mu2: Means of the distributions
            # sigma1, sigma2: Covariance matrices of the distributions
            d = 1/8 * (mu2 - mu1).T @ np.linalg.inv((sigma1 + sigma2) / 2) @ (mu2 - mu1) \
                + 1/2 * np.log(det((sigma1 + sigma2) / 2) / np.sqrt(det(sigma1) * det(sigma2)))
            return d
        
        mean_x1 = np.mean(X1, axis=0)
        mean_x2 = np.mean(X2, axis=0)
        sigma_x1 = np.cov(X1,rowvar=False)
        sigma_x2 = np.cov(X2,rowvar=False)

        overall_bd = bhattacharyya_distance(mean_x1 , sigma_x1, mean_x2, sigma_x2)

        influential_points = []
        for  i in tqdm(range(len(X1))):
            temp_X1 = np.delete(X1, i, axis=0)
            temp_mean = np.mean(temp_X1, axis=0)
            temp_sigma = np.cov(temp_X1, rowvar=False)
            temp_pd = bhattacharyya_distance(temp_mean,temp_sigma , mean_x2, sigma_x2)

            if overall_bd - temp_pd > self.threshold:
                influential_points.append(i)

        return influential_points




# Example usage

# detector = InfluentialPointsDetector()
# X = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
# true_labels = np.array([1, 2, 3, 4, 5])
# predictions = np.array([1.1, 2.5, 2.8, 4.1, 5.2])
# influential_points = detector.detect(X,true_labels, predictions)
# print("Influential Points Indices:", influential_points)


# detector = InfluentialPointsDetector()
# X = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
# true_labels = np.array([1, 2, 3, 4, 5])
# predictions = np.array([1.1, 2.5, 2.8, 4.1, 5.2])
# influential_points = detector.detect(X,true_labels, predictions)
# print("Influential Points Indices:", influential_points)
