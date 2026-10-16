import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relabel_v3 import mention_grades, relabel_value

E, B = "Effusion", "Baker's"
assert mention_grades("Small joint effusion.", E) == ["mild"]
assert mention_grades("Leve derrame articular.", E) == ["mild"]                      # ES
assert mention_grades("Diz eklemi içi sıvı miktarı hafif derecede artmış.", E) == ["mild"]  # TR
assert mention_grades("Manji izljev suprapatelarno.", E) == ["mild"]                 # HR
assert mention_grades("µικρή ποσότητα ενδαρθρικής συλλογής υγρού", E) == ["mild"]   # EL, micro sign
assert mention_grades("Geringer Gelenkerguss.", E) == ["mild"]                       # DE
assert mention_grades("МР данни за минимален ставен излив.", E) == ["mild"]          # BG
assert mention_grades("Minimaal gewrichtsvocht.", E) == ["mild"]                     # NL
assert mention_grades("Il y a un petit épanchement intra-articulaire", E) == ["mild"]  # FR
assert mention_grades("Moderate joint effusion.", E) == ["moderate"]
assert mention_grades("Mild to moderate effusion.", E) == ["moderate"]
assert mention_grades("Kein Gelenkerguss.", E) == []
assert mention_grades("Joint effusion.", E) == ["unspecified"]
assert mention_grades("Kleine Baker-Zyste.", B) == ["mild"]
assert mention_grades("Baker cista promjera 23X9 mm.", B) == ["mild"]
assert mention_grades("Large Baker's cyst.", B) == ["moderate"]

assert relabel_value("Small joint effusion. Small joint effusion.", E, 0.82) == (0.3, "Effusion:0.82->0.3 mild-only")
assert relabel_value("Small effusion. Impression: joint effusion.", E, 0.82)[0] == 0.82  # unspecified mention kept
assert relabel_value("Moderate effusion.", E, 0.95)[0] == 0.95
assert relabel_value("Small effusion.", E, 0.05)[0] == 0.05                         # negatives untouched
print("ok")
