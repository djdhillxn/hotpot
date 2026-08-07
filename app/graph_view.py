import tempfile
import os
import networkx as nx
from pyvis.network import Network

def render_evidence_graph(evidence_graph):
    G = nx.DiGraph()

    for edge in evidence_graph:
        src = edge.get("source", "Start")
        tgt = edge.get("target", "Node")
        lbl = edge.get("label", "")
        G.add_edge(src, tgt, label=lbl)

    net = Network(height="400px", width="100%", bgcolor="#0e1117", font_color="#ffffff", directed=True)
    net.from_nx(G)

    for node in net.nodes:
        if node["id"] == "Question":
            node["color"] = "#ff4b4b"
            node["shape"] = "diamond"
            node["size"] = 25
        else:
            node["color"] = "#00d4b1"
            node["shape"] = "dot"
            node["size"] = 20

    tmp_dir = tempfile.gettempdir()
    html_path = os.path.join(tmp_dir, "evidence_graph.html")
    if hasattr(net, "write_html"):
        net.write_html(html_path)
    elif hasattr(net, "save_graph"):
        net.save_graph(html_path)
    else:
        net.html = net.generate_html()
        with open(html_path, "w") as f:
            f.write(net.html)
    return html_path
