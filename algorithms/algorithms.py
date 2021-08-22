from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope

ALGORITHMS_DICTIONARY = {
    "Kmeans": KMeans(n_clusters=2, random_state=20),
    "Elliptic Envelope": EllipticEnvelope(random_state=10)
}
