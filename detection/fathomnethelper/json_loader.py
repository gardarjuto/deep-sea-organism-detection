import json
from typing import Optional, Any, List
from pathlib import Path
import os


class Taxonomicon:
    def __init__(self):
        root = Path(__file__).parent.absolute()
        with open(os.path.join(root, 'parent_mappings.json'), 'r') as file:
            self.parent_dict = json.load(file)
        with open(os.path.join(root, 'children_mappings.json'), 'r') as file:
            self.children_dict = json.load(file)
        with open(os.path.join(root, 'rank_mappings.json'), 'r') as file:
            self.rank_dict = json.load(file)

    def get_parent(self, concept: str) -> Optional[Any]:
        try:
            return self.parent_dict[concept]
        except KeyError as e:
            print('Concept not found:', e)
            return

    def get_children(self, concept: str) -> Optional[Any]:
        try:
            return self.children_dict[concept]
        except KeyError as e:
            print('Concept not found:', e)
            return

    def get_rank(self, concept: str) -> Optional[Any]:
        try:
            return self.rank_dict[concept]
        except KeyError as e:
            print('Concept not found:', e)
            return

    def get_subtree_nodes(self, concept: str) -> Optional[Any]:
        try:
            nodes = []
            explore = [concept]
            while explore:
                item = explore.pop()
                nodes.append(item)
                for child in self.get_children(item):
                        explore.append(child['name'])
            return nodes
        except KeyError as e:
            print('Concept not found:', e)
            return

    def get_concepts_at_rank(self, rank: str) -> List[str]:
        return [concept for concept in self.rank_dict if self.rank_dict[concept] == rank]
