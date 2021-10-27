from detector.angle_based_outlier_detector import AngleBasedOutlierDetector
from detector.auto_encoder import AutoEncoderDetector
from detector.bayes_change_point import BayesChangePointDetector
from detector.clustering_based_local_outlier_factor import ClusteringBasedLocalOutlierFactorDetector
from detector.contextual_anomaly_detector import ContextualAnomalyDetector
from detector.elliptic_envelope import EllipticEnvelopeDetector
from detector.expose import ExposeDetector
from detector.generalized_esd_test import GeneralizedESDTestDetector
from detector.isolation_forest import IsolationForestDetector
from detector.k_nearest_neighbors import KNearestNeighborsDetector
from detector.knn_cad import KnncadDetector
from detector.local_outlier_factor import LocalOutlierFactorDetector
from detector.one_class_support_vector_machine import OneClassSupportVectorMachineDetector
from detector.principal_component_analysis import PrincipalComponentAnalysisDetector
from detector.relative_entropy import RelativeEntropyDetector
from detector.windowed_gaussian import WindowedGaussianDetector

ALGORITHMS_DICTIONARY = {
    # "Elliptic Envelope": EllipticEnvelopeDetector(),
    # "Isolation Forest": IsolationForestDetector(),
    # "Local Outlier Factor": LocalOutlierFactorDetector(),
    # "Windowed Gaussian": WindowedGaussianDetector(),
    # "Relative Entropy": RelativeEntropyDetector(),
    # "One-class SVM": OneClassSupportVectorMachineDetector(),
    # "Angle-based Outlier Detector": AngleBasedOutlierDetector(),
    "kNN": KNearestNeighborsDetector(),
    "Clustering Based Local Outlier Factor": ClusteringBasedLocalOutlierFactorDetector(),
    "Auto Encoder": AutoEncoderDetector(),
    "KNN CAD": KnncadDetector(),
    # "Expose": ExposeDetector(),
    # "Contextual Anomaly Detector": ContextualAnomalyDetector(),
    # "Bayes Change Point": BayesChangePointDetector(),
}

"""
Algorithms with no anomaly score method, just true or false
"Generalized ESD Test": GeneralizedESDTestDetector(),
    "Principal Component Analysis": PrincipalComponentAnalysisDetector(),

"""