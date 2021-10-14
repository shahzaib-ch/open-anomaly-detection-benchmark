from detector.elliptic_envelope import EllipticEnvelopeDetector
from detector.isolation_forest import IsolationForestDetector
from detector.knncad.knn_cad import KnncadDetector
from detector.local_outlier_factor import LocalOutlierFactorDetector

ALGORITHMS_DICTIONARY = {
    # "Elliptic Envelope": EllipticEnvelopeDetector(),
    # "Isolation Forest": IsolationForestDetector(),
    # "Local Outlier Factor": LocalOutlierFactorDetector(),
    "KNN CAD": KnncadDetector(),
}