from pathlib import Path
import uuid

from classconfig import Config
import pytest

from topicer.base import factory, BaseTopicer
from topicer.tagging.cross_bert import CrossBertTopicer
from topicer.schemas import TextChunk, Tag, TextChunkWithTagSpanProposals


@pytest.fixture(params=["config_local.yaml", "config_hf.yaml"])
def config_path(request):
    return Path(__file__).parent / request.param


@pytest.fixture
def topicer(config_path):
    return factory(config_path)


@pytest.fixture
def sample_inputs():
    text = TextChunk(
        id = uuid.uuid4(),
        text = "lohy a dotace z hlavní pokladny, jeví se příjmův úhrnem 66311 zl. 231/2 kr., z kteréž sumy uděleno chudým 64607 zl. 261/2 kr., pročež zbývá v kase 1753 zl. 961/4 kr. — K stavbě železnice budějovicko-plzenské píše „Bud.“: Jak se nám sděluje, mešká v zdější krajině v obcích venkovských mnoho pracovního lidu cizého, očekávajícího počátek stavby železnice. Lid tento nemá od čeho býti živ a skláda svou nadějí ode dne ke dni v konečné započeti této stavby. Záležitost ta nepokročila však posud dále, než že postavena tento týden nějaka ta bouda a že material zvolna se přiváží. Stanovy záložny v Křinci u Nymburku a čtenář ského spolku ve Větrním Jeníkovu byly od c. k. místodržitelství potvrzeny. Přirážka ažiová činí na západní dráze česke za osoby i dovoz, počinaje od 1. dubna 26 pct. Papírových šestáčků bylo koncem února t. r. v oběhu za 8,398.530 zl. — Zdražení piva. Nejšpatnější aprilový vtip letos provedli pražští sládci, zdraživše z čista jasna pivo o 2 kr. na mázu. Že by ale pivo od včerejška také o 2 kr. lepší bylo, nelze pry pozorovati.",
    )

    tags = [
        Tag(id=uuid.uuid4(), name="Banky a úvěry", description="Informace o spořitelnách, vkladech, úrocích, bankovních produktech, krátkodobých zápůjčkách a hypotékách."),
        Tag(id=uuid.uuid4(), name="Ceny a trh", description="Cenové pohyby a obchodní informace: zdražení zboží (např. piva), burzovní či tržní ceny a obchodní nabídky."),
        Tag(id=uuid.uuid4(), name="Dary a podpora", description="Dary, nadace a sociální pomoc: příspěvky chudým, podporování vdov, stipendia a dobročinné nadace."),
    ]

    return text, tags


def test_init_not_empty(topicer):
    assert topicer is not None


def test_init_correct_cls(topicer):
    assert isinstance(topicer, CrossBertTopicer)


def test_init_value_model(topicer, config_path):
    config = Config(BaseTopicer).load(config_path)
    assert topicer.model == config.untransformed["topicer"]["config"]["model"]


def test_init_value_threshold(topicer, config_path):
    config = Config(BaseTopicer).load(config_path)
    assert topicer.threshold == config.untransformed["topicer"]["config"]["threshold"]


def test_init_value_max_length(topicer, config_path):
    config = Config(BaseTopicer).load(config_path)
    assert topicer.max_length == config.untransformed["topicer"]["config"]["max_length"]


def test_init_value_gap_tolerance(topicer, config_path):
    config = Config(BaseTopicer).load(config_path)
    assert topicer.gap_tolerance == config.untransformed["topicer"]["config"]["gap_tolerance"]


def test_init_value_device(topicer, config_path):
    config = Config(BaseTopicer).load(config_path)
    assert topicer.device == config.untransformed["topicer"]["config"]["device"]


def test_init_value_normalize_score(topicer, config_path):
    config = Config(BaseTopicer).load(config_path)
    assert topicer.normalize_score is config.untransformed["topicer"]["config"]["normalize_score"]


def test_init_value_soft_max_score(topicer, config_path):
    config = Config(BaseTopicer).load(config_path)
    assert topicer.soft_max_score is config.untransformed["topicer"]["config"]["soft_max_score"]
