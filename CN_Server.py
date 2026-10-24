"""
CN Project: Network Design & Analysis System — Analysis Server
================================================================
Integrated Layers:
  - Layer 1: Structural Analysis (Spectral Graph Theory: Laplacian, Fiedler, AP)
  - Layer 2: CN Performance Analysis (Delay, Utilization, Loss, Throughput)
  - Layer 3: AI Random Forest Bottleneck Predictor & Recommender
  - Feature 1: Domain-Based Fitness Analysis (Cost vs Reliability vs Performance Trade-offs)
  - Feature 2: Real-Time Domain Verification (Industry Archetype Benchmark Testing)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import networkx as nx
import numpy as np
from scipy.linalg import eigh
import copy

from ai_engine import ai_recommender

app = Flask(__name__)
CORS(app)


# ─────────────────────────────────────────────
#  Core Math: Spectral Analysis (Layer 1)
# ─────────────────────────────────────────────

def build_graph(nodes, edges):
    """Build a NetworkX graph from frontend data with full QoS link parameters."""
    G = nx.Graph()
    for n in nodes:
        nid = str(n["id"])
        G.add_node(nid, label=n.get("label", nid))
    for e in edges:
        s = str(e["source"])
        t = str(e["target"])
        bw = max(float(e.get("bandwidth", 100.0)), 0.001)
        lat = float(e.get("latency", 5.0))
        traffic = float(e.get("traffic", 20.0))
        
        U = traffic / bw
        d_trans = (12000 / (bw * 1e6)) * 1000.0
        if U < 0.995:
            d_queue = d_trans * (U / (1 - U))
        else:
            d_queue = d_trans * (0.995 / (1 - 0.995))
        d_total = d_trans + lat + 0.1 + d_queue

        G.add_edge(s, t, 
                   hops=1.0, 
                   delay=round(d_total, 3), 
                   bandwidth=bw, 
                   traffic=traffic, 
                   utilization=round(U * 100, 1))
    return G


def laplacian_spectral_analysis(G):
    node_list = list(G.nodes())
    n = len(node_list)

    if n == 0:
        return {}

    A = nx.to_numpy_array(G, nodelist=node_list, weight=None)
    D = np.diag(A.sum(axis=1))
    L = D - A

    eigenvalues, eigenvectors = eigh(L)

    fiedler_value = float(eigenvalues[1]) if n > 1 else 0.0
    lambda_max = float(eigenvalues[-1]) if eigenvalues[-1] != 0 else 1.0
    spectral_gap = fiedler_value / lambda_max if lambda_max != 0 else 0.0

    fiedler_vector = eigenvectors[:, 1] if n > 1 else eigenvectors[:, 0]
    fiedler_map = {node_list[i]: float(fiedler_vector[i]) for i in range(n)}

    eig_vals_A, eig_vecs_A = np.linalg.eigh(A)
    principal_vec = np.abs(eig_vecs_A[:, -1])
    principal_vec = principal_vec / (principal_vec.sum() + 1e-12)
    spectral_centrality = {node_list[i]: float(principal_vec[i]) for i in range(n)}

    betweenness = nx.betweenness_centrality(G, normalized=True)
    weighted_degree = dict(G.degree())
    max_d_val = max(weighted_degree.values()) if weighted_degree else 0
    max_deg = max_d_val if max_d_val > 0 else 1
    normalized_degree = {k: v / max_deg for k, v in weighted_degree.items()}

    articulation_pts = set(nx.articulation_points(G)) if G.number_of_nodes() > 1 else set()

    risk_scores = {}
    for node in node_list:
        sc = spectral_centrality.get(node, 0)
        bc = betweenness.get(node, 0)
        nd = normalized_degree.get(node, 0)
        is_ap = 1.0 if node in articulation_pts else 0.0
        risk = 0.35 * sc + 0.35 * bc + 0.15 * nd + 0.15 * is_ap
        risk_scores[node] = round(risk, 4)

    max_risk = max(risk_scores.values()) if risk_scores else 0
    node_classification = {}
    for node, risk in risk_scores.items():
        is_ap = node in articulation_pts
        if risk >= 0.7 * max_risk and max_risk > 0.05:
            cls = "critical"
        elif risk >= 0.4 * max_risk and max_risk > 0.02:
            cls = "high"
        elif is_ap:
            cls = "high"
        else:
            cls = "normal"
        node_classification[node] = cls

    edge_stress = {}
    for u, v in G.edges():
        stress = (risk_scores.get(u, 0) + risk_scores.get(v, 0)) / 2.0
        edge_stress[f"{u}_{v}"] = round(stress, 4)

    if not nx.is_connected(G):
        connectivity_status = "disconnected"
        health_score = 0.0
    elif fiedler_value < 0.05:
        connectivity_status = "fragile"
        health_score = round(fiedler_value * 5, 3)
    elif fiedler_value < 0.3:
        connectivity_status = "moderate"
        health_score = round(0.25 + fiedler_value * 2, 3)
    else:
        connectivity_status = "robust"
        health_score = min(1.0, round(0.5 + spectral_gap, 3))

    biconnected = list(nx.biconnected_components(G))

    return {
        "fiedler_value": round(fiedler_value, 4),
        "spectral_gap": round(spectral_gap, 4),
        "lambda_max": round(lambda_max, 4),
        "spectral_centrality": {str(k): v for k, v in spectral_centrality.items()},
        "betweenness_centrality": {str(k): round(v, 4) for k, v in betweenness.items()},
        "fiedler_vector": {str(k): round(v, 4) for k, v in fiedler_map.items()},
        "risk_scores": {str(k): v for k, v in risk_scores.items()},
        "node_classification": {str(k): v for k, v in node_classification.items()},
        "edge_stress": edge_stress,
        "articulation_points": [str(a) for a in articulation_pts],
        "connectivity_status": connectivity_status,
        "health_score": health_score,
        "num_biconnected_components": len(biconnected),
        "num_nodes": n,
        "num_edges": G.number_of_edges(),
        "is_connected": nx.is_connected(G)
    }


# ─────────────────────────────────────────────
#  Layer 2 — CN Performance Math
# ─────────────────────────────────────────────

def cn_link_metrics(edge):
    bw = max(float(edge.get("bandwidth", 100.0)), 0.001)
    lat = float(edge.get("latency", 5.0))
    traffic = float(edge.get("traffic", bw * 0.3))

    R_bps = bw * 1e6
    d_trans_ms = (12000 / R_bps) * 1000.0
    d_prop_ms = lat
    d_proc_ms = 0.1

    U = traffic / bw
    U_display = min(U, 1.5)

    if U < 0.995:
        d_queue_ms = d_trans_ms * (U / (1 - U))
        congested = False
    else:
        d_queue_ms = d_trans_ms * (0.995 / (1 - 0.995))
        congested = True

    d_total_ms = d_trans_ms + d_prop_ms + d_proc_ms + d_queue_ms

    if U < 0.8:
        loss_pct = round(0.02 * U * 100, 3)
    elif U < 1.0:
        loss_pct = round(((U - 0.8) / 0.2) * 15.0, 3)
    else:
        loss_pct = 25.0

    throughput_mbps = round(traffic * (1 - loss_pct / 100.0), 4)
    status = "congested" if U >= 0.85 else "high" if U >= 0.65 else "normal"

    return {
        "bandwidth_mbps": bw,
        "latency_ms": lat,
        "traffic_mbps": traffic,
        "d_trans_ms": round(d_trans_ms, 3),
        "d_prop_ms": round(d_prop_ms, 3),
        "d_queue_ms": round(d_queue_ms, 3),
        "d_total_ms": round(d_total_ms, 3),
        "utilization_pct": round(U_display * 100, 1),
        "packet_loss_pct": loss_pct,
        "throughput_mbps": throughput_mbps,
        "status": status,
        "congested": congested
    }


def cn_performance_analysis(G, edges_raw):
    link_results = {}
    for e in edges_raw:
        key = f"{e['source']}_{e['target']}"
        link_results[key] = cn_link_metrics(e)

    if not link_results:
        return {
            "links": {}, "avg_delay_ms": 0, "avg_utilization_pct": 0,
            "avg_packet_loss_pct": 0, "total_throughput_mbps": 0
        }

    delays = [v["d_total_ms"] for v in link_results.values()]
    utils = [v["utilization_pct"] for v in link_results.values()]
    losses = [v["packet_loss_pct"] for v in link_results.values()]
    throughputs = [v["throughput_mbps"] for v in link_results.values()]

    return {
        "links": link_results,
        "avg_delay_ms": round(sum(delays) / len(delays), 2),
        "avg_utilization_pct": round(sum(utils) / len(utils), 1),
        "avg_packet_loss_pct": round(sum(losses) / len(losses), 2),
        "total_throughput_mbps": round(sum(throughputs), 2)
    }


# ─────────────────────────────────────────────
#  Feature 1: Domain-Based Fitness Analysis
# ─────────────────────────────────────────────

def evaluate_domain_fitness(structural, cn, domain):
    """
    Evaluates whether the current network design fits the intended domain use-case.
    Balances Cost vs Reliability vs Performance trade-offs.
    """
    n_nodes = structural.get("num_nodes", 0)
    n_edges = structural.get("num_edges", 0)
    fiedler = structural.get("fiedler_value", 0.0)
    aps = [str(x) for x in structural.get("articulation_points", [])]
    avg_u = cn.get("avg_utilization_pct", 0.0)
    avg_d = cn.get("avg_delay_ms", 0.0)

    # Theoretical edge bounds for n nodes
    min_edges = max(n_nodes - 1, 1)
    max_edges = max((n_nodes * (n_nodes - 1)) // 2, 1)
    edge_density = n_edges / max_edges if max_edges > 0 else 0

    # 1. Cost Index (0 to 100): Higher means more expensive (more cabling, router ports)
    cost_index = min(100, int((n_edges / max_edges) * 75 + (n_edges / max(n_nodes, 1)) * 12))

    # 2. Reliability Index (0 to 100): Higher means more fault-tolerant
    rel_base = min(100, int(fiedler * 140 + (1.0 - (len(aps) / max(n_nodes, 1))) * 50))
    if len(aps) > 0:
        rel_base = min(rel_base, 50)
    reliability_index = max(10, min(100, rel_base))

    # 3. Performance Index (0 to 100): Low delay & balanced load = high performance
    perf_index = max(10, min(100, int(100 - avg_u * 0.4 - min(avg_d, 50) * 0.8)))

    suitability_score = 75
    verdict = ""
    tradeoff_summary = ""
    specific_advice = []

    domain_lower = domain.lower()

    if "campus" in domain_lower or "college" in domain_lower:
        # College Campus: Prioritizes low hardware cost and simplicity.
        # Overkill mesh is flagged as wasteful; Hub/Tree with low cost is favored.
        if cost_index > 55:
            suitability_score = max(35, 95 - int(cost_index * 0.65))
            verdict = "OVER-ENGINEERED (UNECONOMICAL FOR CAMPUS)"
            tradeoff_summary = f"High hardware and port expenditure (Cost Index: {cost_index}/100) across {n_edges} links. While resilience is strong, dense meshing is financially inefficient for university classrooms and labs where 90% of traffic is client-to-server."
            specific_advice.append(f"Cabling & Budget Reduction: Consolidate dense peer links into a Hierarchical Tree or Hub-and-Spoke (~{n_nodes + 1} links) to save budget without degrading student internet access.")
            specific_advice.append("Allocate budget savings to high-speed Wi-Fi 6 access points rather than redundant switch interlinks.")
        elif len(aps) > 0:
            suitability_score = 75 if len(aps) <= 2 else 60
            verdict = "BASIC CAMPUS STAR (BUDGET-OPTIMIZED WITH SERVER UPLINK RISK)"
            tradeoff_summary = f"Cost is low and budget-friendly (Cost Index: {cost_index}/100). The central switch handles client-server traffic well, but {len(aps)} articulation point(s) ({', '.join(aps[:3])}) could isolate lab sectors if that main switch fails."
            specific_advice.append(f"Add a secondary uplink between the main campus router and the server room to ensure academic portals remain online during switch servicing.")
            specific_advice.append("Configure VLAN isolation across campus departments to limit broadcast traffic over the single hub.")
        else:
            suitability_score = 92
            verdict = "OPTIMAL FOR CAMPUS (EXCELLENT COST-TO-PERFORMANCE RATIO)"
            tradeoff_summary = f"Well-balanced star/tree layout (Cost Index: {cost_index}/100) providing cost-effective student distribution with minimal operational complexity."
            specific_advice.append("Current design is economical and perfectly fits university laboratory and lecture hall demands.")

    elif "hospital" in domain_lower or "mission" in domain_lower or "critical" in domain_lower:
        # Mission-Critical / Hospital: 99.999% uptime is mandatory. Single points of failure are FATAL.
        if len(aps) > 0 or not structural.get("is_connected", True):
            suitability_score = 30
            verdict = "NON-COMPLIANT / SEVERE HAZARD (SINGLE FAILURE PARTITION)"
            tradeoff_summary = f"Fatal architectural risk for clinical settings: {len(aps)} Single Point(s) of Failure detected ({', '.join(aps[:3])}). In a hospital, an outage at any of these nodes instantly cuts off ICU patient vitals and MRI imaging systems."
            specific_advice.append(f"MANDATORY HEALTHCARE COMPLIANCE: Eliminate all articulation points by installing active-standby redundant core switches (IEEE 802.1D STP / Cisco Dual-Core Architecture) with dual-homed links to every medical wing.")
            specific_advice.append("Provision isolated emergency power UPS circuits for all core switching gear to ensure uninterrupted telemetry.")
        elif fiedler < 0.25:
            suitability_score = 62
            verdict = "SUB-OPTIMAL (LOW FAULT-TOLERANCE MARGIN)"
            tradeoff_summary = f"Network is connected but lacks robust algebraic connectivity (λ₂ = {fiedler:.3f}). A secondary link cut or switch reboot will sever clinical monitoring telemetry."
            specific_advice.append("Introduce cross-tier redundant links to raise algebraic connectivity (λ₂ > 0.40) to guarantee zero-packet-drop clinical failover.")
        else:
            suitability_score = 96
            verdict = "CLINICAL-GRADE COMPLIANT (HIGH AVAILABILITY DUAL-CORE)"
            tradeoff_summary = f"Outstanding fault tolerance (Reliability Index: {reliability_index}/100, λ₂ = {fiedler:.3f}). Zero single points of failure; fully compliant with healthcare high-availability standards."
            specific_advice.append("Maintain link bandwidth headroom (>30% unused capacity) to absorb telemetry bursts during emergency patient admissions.")

    elif "iot" in domain_lower or "sensor" in domain_lower:
        # IoT Edge: Ultra-low bandwidth, high node count, star/tree to gateway.
        if avg_u < 10 and cost_index > 35:
            suitability_score = 54
            verdict = "MISALIGNED (BANDWIDTH & TOPOLOGY OVER-SPECIFIED)"
            tradeoff_summary = f"Sensor traffic is lightweight and bursty (Avg utilization: {avg_u}%), but hardware cabling is high (Cost Index: {cost_index}/100). Point-to-point wired mesh is uneconomical for low-power sensors."
            specific_advice.append("Consolidate edge sensors around wireless IoT Fog Aggregators (e.g. ZigBee/LoRaWAN/MQTT gateways) rather than point-to-point wired switch ports.")
            specific_advice.append("Right-size link bandwidth to 10 Mbps to minimize edge transceiver energy consumption.")
        elif len(aps) > 0 and len(aps) <= 2:
            suitability_score = 88
            verdict = "TYPICAL IOT HIERARCHICAL EDGE CLUSTER"
            tradeoff_summary = f"Standard IoT star-of-clusters topology. Budget-friendly hardware footprint (Cost Index: {cost_index}/100) with lightweight bandwidth consumption ({avg_u}%)."
            specific_advice.append("Ensure the Fog/Cloud Gateway has adequate queue buffers to absorb periodic synchronized sensor burst transmissions without packet loss.")
        else:
            suitability_score = 84
            verdict = "WELL-SUITED FOR IOT SENSOR COLLECTION"
            tradeoff_summary = f"Cost-effective edge-to-cloud architecture designed for sensor telemetry aggregation without unnecessary cabling overhead."
            specific_advice.append("Verify gateway uplink bandwidth is scaled to handle aggregate telemetry reporting cycles.")

    else: # Default: Enterprise Corporate Office
        # Enterprise: Balanced department segmentation, moderate redundancy, trunk uplinks.
        if len(aps) >= 1:
            suitability_score = 56 if len(aps) == 1 else 46
            verdict = "PARTIAL COMPLIANCE (SINGLE CORE VULNERABILITY)" if len(aps) == 1 else "VULNERABLE (DEPARTMENT ISOLATION RISK)"
            tradeoff_summary = f"Enterprise network contains {len(aps)} critical bottleneck node(s) ({', '.join(aps[:3])}). An outage at these junction(s) will sever inter-department communication between Sales, HR, and Finance."
            specific_advice.append("Deploy redundant dual-trunk distribution switches to eliminate single-point department bottlenecks.")
            specific_advice.append("Implement 802.3ad Link Aggregation (LACP) between access layers and core switches.")
        elif cost_index > 70:
            suitability_score = 64
            verdict = "OVER-PROVISIONED (EXCESSIVE SWITCH PORT COST)"
            tradeoff_summary = f"Full mesh connectivity creates {n_edges} links with Cost Index {cost_index}/100. In an enterprise, 85% of traffic flows north-south to corporate cloud and servers; peer meshing is unnecessary overhead."
            specific_advice.append("Adopt a 3-Tier Hierarchical Access-Distribution-Core design. This retains enterprise resilience while reducing switch port licensing by up to 40%.")
        elif avg_u > 70:
            suitability_score = 62
            verdict = "CONGESTED (CAPACITY UPGRADE REQUIRED)"
            tradeoff_summary = f"Average utilization is {avg_u}%, indicating enterprise video conferences and cloud synchronization are straining access links."
            specific_advice.append("Upgrade trunk uplinks to 1 Gbps and prioritize business-critical traffic using DiffServ QoS tagging.")
        else:
            suitability_score = 92
            verdict = "ENTERPRISE READY (STABLE & BALANCED ARCHITECTURE)"
            tradeoff_summary = f"Solid balance of cost (Cost Index: {cost_index}/100) and operational reliability (Reliability Index: {reliability_index}/100). Fully supports corporate multi-tier business workflows."
            specific_advice.append("Regularly monitor peak core-to-distribution trunk utilization during business hours.")

    return {
        "domain": domain,
        "suitability_score": suitability_score,
        "verdict": verdict,
        "tradeoff_summary": tradeoff_summary,
        "cost_index": cost_index,
        "reliability_index": reliability_index,
        "performance_index": perf_index,
        "recommendations": specific_advice
    }


# ─────────────────────────────────────────────
#  API Routes
# ─────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    domain = data.get("domain", "Enterprise")

    if len(nodes) < 2:
        return jsonify({"error": "Need at least 2 nodes to analyze."}), 400

    G = build_graph(nodes, edges)
    structural = laplacian_spectral_analysis(G)
    cn = cn_performance_analysis(G, edges)

    # Layer 3: AI Random Forest Recommender
    ai_result = ai_recommender.analyze_network(G, structural, cn, edges)

    # Feature 1: Domain-Based Fitness Analysis
    domain_fitness = evaluate_domain_fitness(structural, cn, domain)

    result = {
        "structural": structural,
        "cn_performance": cn,
        "ai_analysis": ai_result,
        "domain_fitness": domain_fitness,
        "network_health": {
            "avg_delay_ms": cn["avg_delay_ms"],
            "avg_utilization_pct": cn["avg_utilization_pct"],
            "avg_packet_loss_pct": cn["avg_packet_loss_pct"],
            "total_throughput_mbps": cn["total_throughput_mbps"],
            "ai_health_score": ai_result["ai_health_score"],
            "ai_network_status": ai_result["ai_network_status"],
            "critical_nodes": sum(1 for c in structural["node_classification"].values() if c == "critical")
        },
        "recommendations": ai_result["ai_recommendations"],
        **structural
    }
    return jsonify(result)


@app.route("/route", methods=["POST"])
def calculate_route():
    data = request.get_json()
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    source = str(data.get("source"))
    target = str(data.get("target"))

    if not source or not target:
        return jsonify({"error": "Both source and target nodes required."}), 400

    G = build_graph(nodes, edges)
    if source not in G or target not in G:
        return jsonify({"error": f"Node {source} or {target} not in graph."}), 404

    if not nx.has_path(G, source, target):
        return jsonify({"has_path": False, "message": "Path is severed."}), 200

    hop_path = nx.shortest_path(G, source=source, target=target, weight="hops")
    hop_delay = sum(G[hop_path[i]][hop_path[i+1]]["delay"] for i in range(len(hop_path)-1))
    hop_max_util = max(G[hop_path[i]][hop_path[i+1]]["utilization"] for i in range(len(hop_path)-1))

    qos_path = nx.shortest_path(G, source=source, target=target, weight="delay")
    qos_delay = sum(G[qos_path[i]][qos_path[i+1]]["delay"] for i in range(len(qos_path)-1))

    route_edges = [f"{qos_path[i]}_{qos_path[i+1]}" for i in range(len(qos_path)-1)]

    if hop_path != qos_path and hop_delay > qos_delay:
        comparison = f"Smart QoS routing chose an alternate {len(qos_path)-1}-hop path ({qos_delay:.1f} ms) over the {len(hop_path)-1}-hop path ({hop_delay:.1f} ms) because the shorter path had a congested link ({hop_max_util}% load) with buffer queuing latency!"
    else:
        comparison = f"Optimal path found: {len(qos_path)-1} hops with {qos_delay:.1f} ms latency. Link buffers have adequate headroom."

    return jsonify({
        "has_path": True,
        "path_nodes": qos_path,
        "path_edges": route_edges,
        "hop_count": len(qos_path) - 1,
        "total_path_delay_ms": round(qos_delay, 2),
        "min_hop_path_nodes": hop_path,
        "min_hop_delay_ms": round(hop_delay, 2),
        "comparison_insight": comparison
    })


@app.route("/simulate_failure", methods=["POST"])
def simulate_failure():
    data = request.get_json()
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    failed_node_id = str(data.get("failed_node", ""))

    G_orig = build_graph(nodes, edges)
    G_fail = copy.deepcopy(G_orig)

    if not failed_node_id or failed_node_id not in G_fail:
        return jsonify({"error": "Failed node not found in graph."}), 400

    failed_node_label = G_orig.nodes[failed_node_id].get("label", f"Node {failed_node_id}")
    neighbors = list(G_orig.neighbors(failed_node_id))
    
    G_fail.remove_node(failed_node_id)
    remaining_nodes = list(G_fail.nodes())

    orig_spectral = laplacian_spectral_analysis(G_orig)
    orig_fiedler = orig_spectral.get("fiedler_value", 0.0)

    if len(remaining_nodes) <= 1:
        is_connected_after = False
        isolated_pairs = []
        fail_fiedler = 0.0
    else:
        is_connected_after = nx.is_connected(G_fail)
        fail_fiedler = laplacian_spectral_analysis(G_fail).get("fiedler_value", 0.0)
        
        isolated_pairs = []
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                u, v = neighbors[i], neighbors[j]
                if u in G_fail and v in G_fail:
                    u_lbl = G_orig.nodes[u].get("label", u)
                    v_lbl = G_orig.nodes[v].get("label", v)
                    if not nx.has_path(G_fail, u, v):
                        isolated_pairs.append(f"{u_lbl} ↔ {v_lbl}")

    if not is_connected_after or len(isolated_pairs) > 0:
        reroute_possible = False
        status = "CRITICAL: NETWORK PARTITIONED (REROUTE FAILED)"
        detail = f"Node '{failed_node_label}' was a Single Point of Failure (Articulation Point)! When it collapsed, communication broke completely between: {', '.join(isolated_pairs[:4])}. No alternate redundant paths exist."
        action = f"REMEDY: Install a redundant bypass link directly connecting {neighbors[0] if len(neighbors)>0 else 'Node A'} and {neighbors[1] if len(neighbors)>1 else 'Node B'}."
    else:
        reroute_possible = True
        status = "RESILIENT REROUTE SUCCESSFUL"
        detail = f"When Node '{failed_node_label}' failed, traffic successfully found alternate redundant paths! However, algebraic connectivity (λ₂) dropped from {orig_fiedler:.3f} to {fail_fiedler:.3f}."
        action = "REMEDY: Monitor remaining links for increased traffic congestion."

    return jsonify({
        "failed_component": failed_node_label,
        "failed_id": failed_node_id,
        "is_connected_after": is_connected_after,
        "reroute_possible": reroute_possible,
        "fiedler_before": orig_fiedler,
        "fiedler_after": fail_fiedler,
        "isolated_pairs": isolated_pairs,
        "status": status,
        "detail": detail,
        "action": action
    })


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "service": "NET-DAS Analysis Server Running"})


if __name__ == "__main__":
    print("=" * 65)
    print("  NET-DAS — Network Design & Analysis System Server")
    print("  http://localhost:5050")
    print("=" * 65)
    app.run(port=5050, debug=False)
