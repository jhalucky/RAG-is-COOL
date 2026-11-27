import sys
import types

from src.agents.planner import parse_planner_response


nodes = types.ModuleType("src.agents.nodes")
nodes.parse_planner_response = parse_planner_response
sys.modules["src.agents.nodes"] = nodes
