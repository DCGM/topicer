import sys
import os
import json
import asyncio
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv
from openai import AsyncOpenAI

# import z parent-parent složky
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from topicer.tagging.config import load_config 

try:
    from topicer.tagging.tag_proposal_v2 import TagProposalV2
    from topicer.schemas import TextChunk, Tag
except ImportError as e:
    print(f"Chyba importu: {e}")
    sys.exit(1)


# --- Validátor ---
def validate_results(test_name, actual_proposals, expected_data):
    """
    Validuje: Počet tagů, Span Start, Span End, Tag ID.
    """
    errors = []

    # seřadíme podle span_start
    actual_sorted = sorted(actual_proposals, key=lambda x: x.span_start)
    expected_sorted = sorted(expected_data, key=lambda x: x['span_start'])

    # kontrola počtu tagů
    if len(actual_sorted) != len(expected_sorted):
        errors.append(
            f"Nesedí počet tagů! Očekáváno: {len(expected_sorted)}, Nalezeno: {len(actual_sorted)}")

    # kontrola položek
    min_len = min(len(actual_sorted), len(expected_sorted))

    for i in range(min_len):
        act = actual_sorted[i]
        exp = expected_sorted[i]

        item_errors = []

        # kontrola span start a span end
        if act.span_start != exp['span_start'] or act.span_end != exp['span_end']:
            item_errors.append(
                f"Pozice: {act.span_start}-{act.span_end} (Očekáváno: {exp['span_start']}-{exp['span_end']})")

        # kontrola tag ID
        if str(act.tag.id) != exp['tag_id']:
            item_errors.append(
                f"Tag ID: {act.tag.id} (Očekáváno: {exp['tag_id']})")

        # chyba
        if item_errors:
            errors.append(
                f"Chyba u položky #{i+1}:\n"
                f"  Actual: {json.dumps(act.model_dump(mode='json'), indent=4, ensure_ascii=False)}\n"
                f"  Expected: {json.dumps(exp, indent=4, ensure_ascii=False)}\n"
                f"  Detail: " + "; ".join(item_errors)
            )

    if errors:
        print(f"   ❌ TEST FAILED: {test_name}")
        for err in errors:
            print(f"      - {err}")
        return False
    else:
        print(f"   ✅ TEST PASSED: {test_name}")
        return True


async def run_tests():
    load_dotenv(project_root / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Chyba: OPENAI_API_KEY není nastaven v .env souboru")
        exit(1)

    inputs_dir = current_dir / "inputs"
    outputs_dir = current_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    client = AsyncOpenAI(api_key=api_key)
    
    # Load the AppConfig object correctly
    config_obj = load_config() 
    
    # Initialize TagProposalV2 with the AppConfig object
    proposal_service = TagProposalV2(config_obj, client) 

    files = list(inputs_dir.glob("*.json"))
    total_tests = 0
    passed_tests = 0

    for input_file in files:
        total_tests += 1
        test_name = input_file.stem
        print(f"Test: {test_name}")

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Příprava
            chunk = TextChunk(id=uuid4(), text=data["text"])
            tags_objects = [Tag(**t) for t in data["tags"]]
            expected = data.get("expected", None)

            # Volání funkce
            result = await proposal_service.propose_tags(chunk, tags_objects)

            # Uložení výstupu
            output_path = outputs_dir / f"{test_name}_output.json"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=4, ensure_ascii=False))

            # Validace
            if expected is not None:
                if validate_results(test_name, result.tag_span_proposals, expected):
                    passed_tests += 1
            else:
                print(f"   SKIP - no 'expected' in input")

        except Exception as e:
            print(f"   ERROR: {e}")

        print("-" * 60)

    print(f"\nVýsledek: {passed_tests}/{total_tests} testů.")
    await client.close()

if __name__ == "__main__":
    asyncio.run(run_tests())
