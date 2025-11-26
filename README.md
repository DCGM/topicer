# topicer

## windows setup

python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt

## spuštění s debug flagem

`python ./tag_proposals.py`

## spuštění bez debug flagu

`python ./tag_proposals.py -O`

## formátování

pro formátování využíváme autopep8

## testování funkce propose_tags2

`python tests/propose_tags2/script.py`
