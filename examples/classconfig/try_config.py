#!/usr/bin/env python3
from abc import ABC

from classconfig import CreatableMixin, ConfigurableMixin, ConfigurableSubclassFactory, Config, ConfigurableValue, \
    RelativePathTransformer


class TopicerMethod(ABC):
    ...


class MyTopicerMethod(TopicerMethod, ConfigurableMixin):
    n_topics: int = ConfigurableValue(user_default=10, desc="Number of topics to generate")
    model_path: str = ConfigurableValue(desc="Path to the model file", user_default="my_model.bin", transform=RelativePathTransformer())

    def generate_topics(self, data: list[str]) -> list[str]:
        return data


class AnotherTopicerMethod(TopicerMethod, ConfigurableMixin):
    n_topics: int = ConfigurableValue(user_default=5, desc="Number of topics to generate", validator=lambda x: x > 0)
    topics: list[str] = ConfigurableValue(user_default=["Topic A", "Topic B", "Topic C", "Topic D", "Topic E"],
                                         desc="Predefined topics")

    def __post_init__(self):
        pass
    def generate_topics(self, data: list[str]) -> list[str]:
        return [self.topics[i % len(self.topics)] for i in range(len(data))]


class TopicerMethodFactory(CreatableMixin, ConfigurableMixin):
    method: TopicerMethod = ConfigurableSubclassFactory(TopicerMethod,
                                                        desc="Method that will be used for obtaining topics.",
                                                        user_default=MyTopicerMethod)


def factory(cfg: str | dict | Config):
    return TopicerMethodFactory.create(cfg).method


config_path = "config.yaml"

conf = Config(TopicerMethodFactory)
conf.save(config_path)
conf.to_md("configuration_doc.md")


obj = factory(config_path)

data = obj.generate_topics(["This is a sample document.", "Another document goes here."])
print(data)

print(obj.model_path)
print(type(obj))
