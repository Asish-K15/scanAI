SUPPORTED_SPECIES = {"dog", "cat", "cattle"}


def route_species(species: str) -> str:
    species = species.strip().lower()

    if species not in SUPPORTED_SPECIES:
        raise ValueError(
            f"Unsupported species: {species}. "
            f"Supported species: {sorted(SUPPORTED_SPECIES)}"
        )

    return species
