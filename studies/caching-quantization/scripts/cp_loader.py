"""Load chemprop's real atom/bond/molgraph featurizers without the full package deps."""
import importlib.util, sys, types
from enum import StrEnum

ROOT = "/Users/svaddadi/Documents/GitHub/chemprop/chemprop"

for name in ("chemprop", "chemprop.featurizers", "chemprop.featurizers.molgraph",
             "chemprop.utils", "chemprop.data"):
    mod = types.ModuleType(name); mod.__path__ = []; sys.modules[name] = mod

# minimal real EnumMapping (verbatim semantics from chemprop/utils/utils.py)
u = types.ModuleType("chemprop.utils.utils")
class EnumMapping(StrEnum):
    @classmethod
    def get(cls, name):
        if isinstance(name, cls):
            return name
        try:
            return cls[name.upper()]
        except KeyError:
            raise KeyError(f"unsupported value {name}")
    @classmethod
    def keys(cls):
        return (e.name.lower() for e in cls)
u.EnumMapping = EnumMapping
sys.modules["chemprop.utils.utils"] = u

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_load("chemprop.data.molgraph", f"{ROOT}/data/molgraph.py")
_load("chemprop.featurizers.base", f"{ROOT}/featurizers/base.py")
atom = _load("chemprop.featurizers.atom", f"{ROOT}/featurizers/atom.py")
bond = _load("chemprop.featurizers.bond", f"{ROOT}/featurizers/bond.py")
_load("chemprop.featurizers.molgraph.mixins", f"{ROOT}/featurizers/molgraph/mixins.py")

# molgraph/molecule.py hard-imports cuik_molmaker and torch; stub cuik only
c = types.ModuleType("cuik_molmaker")
for n in ("atom_onehot_feature_names_to_array", "atom_float_feature_names_to_array",
          "bond_feature_names_to_array", "batch_mol_featurizer"):
    setattr(c, n, lambda *a, **k: None)
sys.modules["cuik_molmaker"] = c
molecule = _load("chemprop.featurizers.molgraph.molecule", f"{ROOT}/featurizers/molgraph/molecule.py")

SimpleMoleculeMolGraphFeaturizer = molecule.SimpleMoleculeMolGraphFeaturizer
MultiHotAtomFeaturizer = atom.MultiHotAtomFeaturizer
MultiHotBondFeaturizer = bond.MultiHotBondFeaturizer
