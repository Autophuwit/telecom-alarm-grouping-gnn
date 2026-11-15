"""alarm-incident-gnn: group network alarms into incidents with a graph neural network.

Pipeline, one day at a time:
    data.load_corpus      read alarms, tickets and inventory
    graph.build_day       candidate alarm pairs and their features
    labels.Labeler        weak pair labels from clear times, weighted by how likely they are real
    model.PairGNN         GATv2 encoder + symmetric pair head
    cluster               average-linkage clustering, then tickets join clusters into incidents
    evaluate              metrics against the simulator's ground truth
"""
__version__ = "0.1.0"
