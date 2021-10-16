from detector.bayes_change_point import BayesChangePointDetector
from detector.contextual_anomaly_detector import ContextualAnomalyDetector
from detector.expose import ExposeDetector
from detector.knn_cad import KnncadDetector
from detector.relative_entropy import RelativeEntropyDetector
from detector.windowed_gaussian import WindowedGaussianDetector

ALGORITHMS_DICTIONARY = {
    # "Elliptic Envelope": EllipticEnvelopeDetector(),
    # "Isolation Forest": IsolationForestDetector(),
    # "Local Outlier Factor": LocalOutlierFactorDetector(),
    # "KNN CAD": KnncadDetector(),
    # "Windowed Gaussian": WindowedGaussianDetector(),
    # "Bayes Change Point": BayesChangePointDetector(),
    # "Expose": ExposeDetector(),
   # "Contextual Anomaly Detector": ContextualAnomalyDetector(),
    "Relative Entropy": RelativeEntropyDetector(),
}