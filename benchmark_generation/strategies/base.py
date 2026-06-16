from abc import ABC, abstractmethod


class BenchmarkStrategy(ABC):

    @abstractmethod
    def generate_question(
        self,
        llm,
        model,
        chunk
    ):
        pass

    @abstractmethod
    def generate_answer(
        self,
        llm,
        model,
        question,
        chunk
    ):
        pass

    @property
    def name(self):
        return self.__class__.__name__