import inspect
from typing import Union, Dict, Any


def get_all_metrics():
    from ad_baseline import metrics as library
    from ad_baseline.metrics.base import AnomalyMetric

    metrics = dict()
    for name, cls_instance in library.__dict__.items():
        if inspect.isclass(cls_instance) and \
                issubclass(cls_instance, AnomalyMetric) and \
                (cls_instance != AnomalyMetric):
            instance = cls_instance.from_config()
            metrics[instance.name()] = instance
    return metrics


def get_metric(instance: Union["AnomalyMetric", Dict[str, Any], str]):
    from ad_baseline.metrics.base import AnomalyMetric
    if isinstance(instance, AnomalyMetric):
        return instance

    if isinstance(instance, str):
        instance = dict(name=instance)

    instance_name = instance.pop('name', None)

    if instance_name is None:
        raise ValueError(f"Error: instance name is not specified!")

    from ad_baseline import metrics as library

    for name, cls_instance in library.__dict__.items():
        if inspect.isclass(cls_instance) and \
                issubclass(cls_instance, AnomalyMetric) and \
                (cls_instance.name() != instance_name):
            return cls_instance.from_config(**instance)

    raise ValueError(f"Error: unknown instance name {instance_name}!")
