"""
Base agent class defining the agent lifecycle.
"""
try:
    from abc import ABC, abstractmethod
except ImportError:
    # Python 2.7 compatibility
    from abc import ABCMeta, abstractmethod
    ABC = object

try:
    from typing import Any, Dict
except ImportError:
    # Python 2.7 compatibility
    Any = object
    Dict = dict


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the forecast monitoring system.
    
    Defines the standard agent lifecycle: observe -> reason -> act
    """
    
    def __init__(self):
        """Initialize the base agent."""
        self._data = None
        self._reasoning_result = None
    
    @abstractmethod
    def observe(self, data):
        """
        Observe and store input data.
        
        Args:
            data: Input data to be processed by the agent
        """
        pass
    
    @abstractmethod
    def reason(self):
        """
        Process the observed data and perform reasoning.
        
        This method should analyze the data stored during observe()
        and prepare results for the act() method.
        """
        pass
    
    @abstractmethod
    def act(self):
        """
        Return the result of the agent's processing.
        
        Returns:
            The final output/result of the agent's analysis
        """
        pass
    
    def reset(self):
        """
        Reset the agent's internal state for processing new data.
        
        This method clears any stored data and reasoning results,
        allowing the agent to be reused for multiple processing cycles.
        """
        self._data = None
        self._reasoning_result = None