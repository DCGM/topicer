import json
import argparse
import requests


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="TopicerAPI URL.")

    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("--list-configs", help="List available Topicer configs.", action="store_true")
    group1.add_argument("--config-name", help="Name of the Topicer config to use.")

    parser.add_argument("--method", help="Method to use.",
                        choices=["discover_topics_sparse", "discover_topics_dense", "discover_topics_in_db_sparse",
                                 "discover_topics_in_db_dense", "propose_tags", "propose_tags_in_db"])

    parser.add_argument("--text-chunk", help="Path to JSON file with TextChunk.",
                        required=False, default=None)
    parser.add_argument("--tags", help="Path to JSON file with list of Tags.",
                        required=False, default=None)
    parser.add_argument("--text-chunks", help="Path to JSON file with list of TextChunks.",
                        required=False, default=None)
    parser.add_argument("--db-request", help="Path to JSON file with DBRequest.",
                        required=False, default=None)
    parser.add_argument("--tag", help="Path to JSON file with Tag.",
                        required=False, default=None)
    parser.add_argument("--n", help="Number of topics to discover (for discover_topics methods",
                        type=int, required=False, default=None)

    parser.add_argument("--output", help="Path to the output file.", required=False, default=None)

    args = parser.parse_args()
    return args


def compose_url(args) -> str:
    methods_mapping = {
        "discover_topics_sparse": "topics/discover/texts/sparse",
        "discover_topics_dense": "topics/discover/texts/dense",
        "discover_topics_in_db_sparse": "topics/discover/db/sparse",
        "discover_topics_in_db_dense": "topics/discover/db/dense",
        "propose_tags": "tags/propose/texts",
        "propose_tags_in_db": "tags/propose/db",
    }

    url = f"{args.url.rstrip('/')}/v1/{methods_mapping[args.method]}?config_name={args.config_name}"

    n_parameter_methods = ["discover_topics_sparse", "discover_topics_dense", "discover_topics_in_db_sparse",
                           "discover_topics_in_db_dense"]

    if args.method in n_parameter_methods and args.n is not None:
        url += f"&n={args.n}"

    return url


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_payload(args) -> dict:
    if args.method in ["discover_topics_sparse", "discover_topics_dense"]:
        if args.text_chunks is None:
            raise ValueError("Argument --text-chunks must be provided for discover_topics_sparse/dense methods.")

        text_chunks = load_json(args.text_chunks)

        payload = text_chunks

    elif args.method in ["discover_topics_in_db_sparse", "discover_topics_in_db_dense"]:
        if args.db_request is None:
            raise ValueError("Argument --db-request must be provided for discover_topics_in_db_sparse/dense methods.")

        db_request = load_json(args.db_request)

        payload = db_request

    elif args.method == "propose_tags":
        if args.text_chunk is None or args.tags is None:
            raise ValueError("Both --text-chunk and --tags must be provided for propose_tags method.")

        text_chunk = load_json(args.text_chunk)
        tags = load_json(args.tags)

        payload = {"text_chunk": text_chunk, "tags": tags}

    elif args.method == "propose_tags_in_db":
        if args.db_request is None or args.tags is None:
            raise ValueError("Both --db-request and --tag must be provided for propose_tags_in_db method.")

        db_request = load_json(args.db_request)
        tag = load_json(args.tag)

        payload = {"db_request": db_request, "tag": tag}

    else:
        raise NotImplementedError("Not implemented yet.")

    return payload

def send_request(url: str, payload: dict) -> dict:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def process_response(response: dict, args) -> None:
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2, ensure_ascii=False)
    else:
        if args.method in ["discover_topics_sparse", "discover_topics_dense", "discover_topics_in_db_sparse", "discover_topics_in_db_dense"]:
            from topicer.schemas import DiscoveredTopics, DiscoveredTopicsSparse

            if args.method in ["discover_topics_sparse", "discover_topics_in_db_sparse"]:
                result = DiscoveredTopicsSparse(**response)
                print("Discovered Topics (Sparse):")
            else:
                result = DiscoveredTopics(**response)
                print("Discovered Topics (Dense):")

            print("-----")
            for i, topic in enumerate(result.topics):
                print(f"Topic: {topic.name}")

                if topic.name_explanation:
                    print(f"Explanation: {topic.name_explanation}")

                if topic.description:
                    print(f"Description: {topic.description}")

                if i < len(result.topics) - 1:
                    print("-----")

        elif args.method in ["propose_tags", "propose_tags_in_db"]:
            from topicer.schemas import TextChunkWithTagSpanProposals

            if args.method == "propose_tags":
                results = [TextChunkWithTagSpanProposals(**response)]
            else:
                results = [TextChunkWithTagSpanProposals(**item) for item in response]

            for i, result in enumerate(results):
                print("Text:", result.text)
                print("-----")
                for j, tag_span_proposal in enumerate(result.tag_span_proposals):
                    print(f"Tag: {tag_span_proposal.tag.name}")
                    print(f"Span: ({tag_span_proposal.span_start}, {tag_span_proposal.span_end})")
                    print(f"Proposed text: '{result.text[tag_span_proposal.span_start:tag_span_proposal.span_end]}'")
                    print(f"Confidence: {tag_span_proposal.confidence:.4f}")

                    if i < len(results) - 1 or j < len(result.tag_span_proposals) - 1:
                        print("-----")

        else:
            print(json.dumps(response, indent=2, ensure_ascii=False))


def list_configs(args) -> None:
    url = f"{args.url.rstrip('/')}/v1/configs"

    response = requests.get(url)
    response.raise_for_status()
    configs = response.json()

    print("Available Topicer configs:")
    for config in configs:
        print(f"- {config}")

def main():
    args = parse_arguments()

    if args.list_configs:
        list_configs(args)
    else:
        if args.method is None:
            raise ValueError("Argument --method must be provided when --config-name is specified.")

        url = compose_url(args)
        payload = prepare_payload(args)
        response = send_request(url, payload)
        process_response(response, args)

    return 0


if __name__ == "__main__":
    exit(main())
