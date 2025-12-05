import pytest
from topicer.utils.template import Template, TemplateTransformer


@pytest.fixture
def transformer():
    return TemplateTransformer()


def test_template_render():
    template = Template("Hello {{ name }}!")
    rendered = template.render({"name": "World"})
    assert rendered == "Hello World!"


# 2. Inject the fixture into the test function
def test_transformer_transform(transformer):
    template = transformer("Hello {{ name }}!")
    rendered = template.render({"name": "World"})
    assert rendered == "Hello World!"