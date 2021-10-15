from detector.bayes_change_point import BayesChangePointDetector
from detector.knn_cad import KnncadDetector
from detector.windowed_gaussian import WindowedGaussianDetector

ALGORITHMS_DICTIONARY = {
    # "Elliptic Envelope": EllipticEnvelopeDetector(),
    # "Isolation Forest": IsolationForestDetector(),
    # "Local Outlier Factor": LocalOutlierFactorDetector(),
    # "KNN CAD": KnncadDetector(),
    # "Windowed Gaussian": WindowedGaussianDetector(),
    "Bayes Change Point": BayesChangePointDetector(),
}