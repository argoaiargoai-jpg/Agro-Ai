"""Specialist plant / disease providers used BEFORE the generative step, when our own model is not confident enough.

PlantIdentifier  -> which plant is this?        (PlantNetProvider)
DiseaseDiagnoser -> which condition is this?    (PlantixProvider, KindwiseProvider = crop.health / plant.health)
All are optional: a provider without a key is skipped, a failing provider never breaks the analysis. Customers never see provider names.
"""
