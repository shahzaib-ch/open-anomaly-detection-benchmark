from detector.bayes_change_point import BayesChangePointDetector
from detector.contextual_anomaly_detector import ContextualAnomalyDetector
from detector.expose import ExposeDetector
from detector.generalized_esd_test import GeneralizedESDTestDetector
from detector.knn_cad import KnncadDetector
from detector.one_class_support_vector_machine import OneClassSupportVectorMachineDetector
from detector.principal_component_analysis import PrincipalComponentAnalysisDetector
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
    # "Relative Entropy": RelativeEntropyDetector(),
    # "Generalized ESD Test": GeneralizedESDTestDetector(),
    # "Principal Component Analysis": PrincipalComponentAnalysisDetector(),
    "One-class SVM": OneClassSupportVectorMachineDetector(),
}