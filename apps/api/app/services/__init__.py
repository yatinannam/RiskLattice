"""Services package for the RiskLattice API.

The API is orchestration/serialization only. All risk logic stays in the
engine layer. Services load the dataset, build model predictions, the graph,
campaign assessments, containment recommendations and investigator evidence
once at startup, then serve the prepared state from memory.
"""