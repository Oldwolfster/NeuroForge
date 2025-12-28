from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Union


class BaseArena(ABC):
    """
    Abstract base class for different types of training data TrainingPits.
    """
    @abstractmethod
    def generate_training_data(self) -> List[Tuple[Any, ...]]:

        pass
    def generate_training_data_with_or_without_labels(self) -> Union[List[Tuple[Any, ...]], Tuple[List[Tuple[Any, ...]], List[str]]]:
        """
        Handle labels if they exist, otherwise return only the data.
        """
        if hasattr(self, 'labels'):  # Check if the subclass has a `labels` attribute
            return self.generate_training_data(), self.labels
        return self.generate_training_data()