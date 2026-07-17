# Client USER VIEW filter — Z5 closure audit

> Post-audit TODO #1 z `docs/data_source_audit.md` (Fragment H, D-H.7).
>
> **Cel:** zmapować gdzie w kliencie sto-warp filtrujemy klasy wirtualne
> (`__empty__`, `__inactive__`, `__boff_*`) na **wyjściu** rozpoznania
> (k-NN seed, recognition results, build planner output, UI render),
> żeby bezpiecznie zamknąć **Z5** od strony klienta i odblokować **D-B.3**
> (komentowanie filtra na *wejściu* w `sync_client.py:434-436`).
>
> Status: closure dla Z5 — pokrycie input (D-A.1) + output (ten dokument).

---

## 1. Z5 — pełne brzmienie

Z `docs/data_source_audit.md` §2:

> **Klasy wirtualne (`__*`) są legalne dla ML, ale nie dla seedingu k-NN
> usera.** ML musi je rozróżniać; user dostaje tylko cropy itemów z gry.
>
> Konsekwencja: trzeba mieć rozróżnienie: dane ML mogą zawierać `__*`,
> dane „cropy dla usera" nie.

Po **D-A.1** `data/` jest jednym wspólnym zbiorem (ML + k-NN seed klienta).
Z5 musi więc być realizowane w **filtrach na wyjściu** w kliencie:
filtrowanie nie kiedy `__*` wchodzi do `data/`, tylko kiedy `data/` jest
konsumowany do prezentacji/użycia po stronie usera.

---

## 2. Mapa filtrów `__*` w kliencie

Grep `__empty__|__inactive__|__boff_|startswith('__'` w `warp/`
(z wyłączeniem trenera produkującego modele, gdzie virtual to legalna
klasa ML).

### 2.1 RECOGNITION INPUT — `recognition/icon_matcher.py`

Te filtry chronią Stage 0 (pHash override z `knowledge.json`) przed
zatrutymi wpisami w knowledge.json — to **defense-in-depth na poziomie
modelu**, niezależne od kontroli backendu.

| Linia | Co robi | Klasa filtra |
|---|---|---|
| 244 | Suppression Stage 0 pHash override jeśli `name.startswith('__')` lub `Test Item Name` — odrzuca fallback do ML/template | input → recognizer |
| 260-267 | Embedder cross-check w Stage 0: jeśli pHash override mówi „X", ale embedder mówi `__*` z conf ≥ 0.40 → suppress override | input → recognizer |
| 907-916 | `seed_from_training_data`: skip jeśli virtual label + colourful crop (poison) | seed → ML session pool |
| 996-1001 | `seed_from_community_crops`: skip jeśli virtual label + colourful crop (poison) | seed → ML session pool |

**Uwaga:** seedery (907, 996) NIE filtrują virtual jako klasy — filtrują
„virtual label + colourful crop" jako *mismatch* (typowy poison: real ikona
mislabeled jako empty/inactive). Prawdziwe `__empty__` crops (dim, uniform)
**są seedowane** jako legalne session examples — żeby anti-virtual-bias
combine stage miał kontekst dla zwycięstwa real-icon nad virtual-session.

### 2.2 RECOGNITION OUTPUT — `recognition/icon_matcher.py` (combine stage)

Logika **„anti-virtual-bias"** w combine stage (linie 313-410) — to jest
**rdzeń ochrony Z5 dla usera**. Trzy reguły suppress virtual:

| Linia | Reguła | Sytuacja |
|---|---|---|
| 321-323 | `ml_real = ml_name not startswith('__') and ml_conf ≥ 0.40` → ML wins | ML mówi real icon — virtual session/template są tłumione |
| 348-350 | `sess_virtual_perfect = virtual + sess_score ≥ 0.95` + ML mówi real z conf ≥ 0.15 → poison guard | Self-match z poisoned training data — kill virtual |
| 336-356 | `query_looks_real = bright > 0.15 AND rich > 0.15` + sess/template wirtualne → kill virtual | Query crop bright + colourful, virtual logicznie niemożliwy |

Output ostateczny: jeśli żadna z trzech reguł nie zadziała, virtual może
wygrać i wrócić jako recognition result do `warp_importer`. Wtedy filter
przechodzi do warstwy 2.3/2.4.

| Linia | Co robi | Klasa filtra |
|---|---|---|
| 404-405 | `_thumb_for_name`: returns None dla virtual (`startswith('__')`) | UI presentation |

### 2.3 OUTPUT TO BUILD PLANNER — `warp_importer.py` + `build_writer.py`

To są **konsumenci recognition results** którzy zapisują build do SETS
(jeśli bridge aktywny). Virtual NIE może trafić do build planner —
filter MUSI zatrzymać go na tym poziomie.

| Plik:linia | Co robi |
|---|---|
| `warp_importer.py:58` | `VIRTUAL_ITEM_NAMES = frozenset({'__empty__', '__inactive__'})` |
| `warp_importer.py:182-184` | Confidence weighting: virtual przy low-conf liczy się jako 0.5× — depriorytetyzacja |
| `warp_importer.py:2073` | Mapping `cell_state` ('empty'/'inactive') → vname (`__empty__`/`__inactive__`) — translation layer |
| `warp_importer.py:2261` | `if name not in ('__empty__', '__inactive__')` — skip virtual w jakiejś agregacji |
| `warp_importer.py:2576-2598` | BOFF profession resolution: virtual → seat's typed profession (label transform) |
| `warp_importer.py:2968` | `names_set.update(VIRTUAL_ITEM_NAMES)` — virtual jako legalne w kontekście możliwych nazw |
| `warp_importer.py:2979` | `if item_name in VIRTUAL_ITEM_NAMES: ...` — handling virtual w slot |
| `build_writer.py:39` | `VIRTUAL_ITEM_NAMES = frozenset({'__empty__', '__inactive__'})` |
| `build_writer.py:231` | `if not ri.name or ri.name in VIRTUAL_ITEM_NAMES: continue` — SKIP zapis do build |
| `build_writer.py:455` | Count „active" slots: `if ri.name and ri.name not in VIRTUAL_ITEM_NAMES` |
| `build_writer.py:573` | BOFF slot handling: virtual = empty slot dla rank check |

**Kluczowy:** `build_writer.py:231` — to jest **ostateczny filter dla build
output**. Virtual nigdy nie trafia jako nazwa itemu do build planner.

### 2.4 TRAINER UI RENDER — `trainer/trainer_window.py`

Trener (WARP CORE) wyświetla wyniki rozpoznania userowi do confirmation.
Virtual NIE może być pokazane jako "`__empty__`" — musi być human-friendly.

| Linia | Co robi |
|---|---|
| 1998-2019 | Render: jeśli `is_virtual`, pokaż `'[empty slot]'` / `'[inactive slot]'` zamiast literalnej nazwy |
| 2864 | `elif name not in VIRTUAL_ITEM_NAMES` — gating jakiejś walidacji nie-virtualem |
| 3629 | `if name in VIRTUAL_ITEM_NAMES and ri.get('src') == 'session': ...` — never auto-accept virtual z session crop |
| 3727 | `set(self._build_search_candidates(slot)) \| set(VIRTUAL_ITEM_NAMES)` — search candidates include virtuals jako wybór dla manual labeling |
| 4011 | `if not name or name in VIRTUAL_ITEM_NAMES: ...` — skip virtual w jakiejś agregacji |
| 4135, 4301 | `for vname in sorted(VIRTUAL_ITEM_NAMES): ...` — dodawanie virtuali do labelers (manual selection) |

W trenerze virtual ma **dwa rola**:
- (a) wyświetlany jako human-friendly `[empty slot]` (UX, nie filter).
- (b) dostępny jako manual label dla user-confirmation („tak, to jest empty").

Bez tego user nie mógłby corectly oznaczyć empty slotów do treningu ML.

### 2.5 BOFF KEYS — `recognition/boff_keys.py`

| Linia | Co robi |
|---|---|
| 158 | `_VIRTUAL_NAMES = frozenset({'', '__empty__', '__inactive__'})` |
| 190 | Filter: `if _field(it, 'name', '') not in _VIRTUAL_NAMES` — collect BOFF abilities POMIJAJĄC virtual |

To OUTPUT filter — BOFF ability sheet dla usera nie zawiera virtual.

### 2.6 LAYOUT DETECTOR — `recognition/layout_detector.py`

Generator `__boff_<prof>` labels. To NIE jest filter, to **produkcja** tych
specjalnych virtual labels jako kontekst dla detekcji. Virtual labels dla
slotów BOFF dają informację „tu może siedzieć Tac/Eng/Sci ability".

Linie 288, 305-306, 2436: generate `__boff_<prof>`.
Linie 417, 2212, 2890: handle `__empty__` jako expected outcome (informational).

### 2.7 KNOWLEDGE LAYER (sync_client) — INPUT FILTER

**Jedyny INPUT-side filter w kliencie**, do zlikwidowania per D-B.3:

| Linia | Co robi |
|---|---|
| `knowledge/sync_client.py:434-436` | `if _name.startswith('__') or _name == 'Test Item Name': return` — filter przy contribute knowledge |

Ten filter NIE chroni usera; chroni `knowledge.json` przed virtual hard-override.
Po **D-A.1** chcemy żeby `__*` mogło trafić do `knowledge.json` jako legalne,
a Stage 0 (`icon_matcher.py:244`) ma **defense-in-depth** który je tam i tak
odrzuci jako hard-override. Zatem `sync_client.py:434-436` jest **redundantny**.

### 2.8 SCRUB / CONFLICT REVIEW TOOLS — maintainer tools

| Plik:linia | Co robi |
|---|---|
| `tools/scrub_training_data.py:44-147` | Offline poison scrub — visual sanity check dla virtual labels |
| `tools/conflict_reviewer.py:47-56` | Conflict review — virtual identification |

Te tools są **maintainer-only**, nie wpływają na user view runtime.

---

## 3. Klasyfikacja filtrów wg Z5

**Filtry CHRONIĄCE USER VIEW** (Z5 closure od strony wyjścia):

1. **Stage 0 defense-in-depth** (`icon_matcher.py:244, 260-267`)
   — chroni przed zatrutym `knowledge.json` na poziomie pHash override.
2. **Anti-virtual-bias combine** (`icon_matcher.py:321-356`)
   — trzy reguły suppress virtual w final recognition result.
3. **Seed poison guard** (`icon_matcher.py:907, 996`)
   — chroni session pool przed self-match na poisoned crops.
4. **`_thumb_for_name` None dla virtual** (`icon_matcher.py:404`)
   — UI nie pokazuje thumbnail virtual.
5. **`build_writer.py:231, 455, 573`**
   — virtual nie ląduje w build planner.
6. **`warp_importer.py:182, 2576-2598`**
   — confidence weighting + BOFF profession resolution.
7. **`boff_keys.py:190`**
   — BOFF ability sheet pomija virtual.
8. **Trainer UI render** (`trainer_window.py:1998-2019`)
   — human-friendly '[empty slot]' zamiast `__empty__`.

**Filtry NIE-USER-VIEW** (do likwidacji per D-B.3 lub legalnie internal):

- `sync_client.py:434-436` — input filter dla knowledge contribution, redundantny.
- `trainer/training_data.py:88`, `trainer/embedder_trainer.py:102` — ML training data, virtual LEGALNE per D-A.1.
- `layout_detector.py:288, 305, 2436` — PRODUKCJA `__boff_*` labels, internal.
- `tools/scrub_training_data.py`, `tools/conflict_reviewer.py` — maintainer-only.

---

## 4. Decyzje Z5 closure

| # | Decyzja | Notatka |
|---|---------|---------|
| Z5-C.1 | **Wielowarstwowa obrona Z5 od strony wyjścia istnieje i jest wystarczająca.** 8 punktów filtra wymienionych w §3 zapewnia że żaden `__*` nie dotrze do build plannera (build_writer.py:231), thumbnail UI (icon_matcher.py:404), BOFF sheet (boff_keys.py:190) ani recognition result na real-looking icon (icon_matcher.py:321-356 anti-virtual-bias). | Z5 zamknięte od wyjścia. |
| Z5-C.2 | **`sync_client.py:434-436` może być bezpiecznie zakomentowany (D-B.3).** Defense-in-depth Stage 0 (`icon_matcher.py:244`) złapie `__*` z `knowledge.json` jako hard-override suppression — nawet jeśli backend wpisze tam virtual. Redundancja jest zamierzona (D-A.1 spec). | Odblokowanie D-B.3. |
| Z5-C.3 | **Komentowanie filtra w `sync_client.py` musi być symetryczne z komentowaniem w backendzie (D-A.1 + D-G.1).** Nie wolno wyłączyć tylko jednego — albo oba (klient + backend), albo żaden. Rollback jednoczesny. | Spójność rollback. |
| Z5-C.4 | **Anti-virtual-bias thresholds nie powinny być modyfikowane bez evidence z combat testów.** VIRTUAL_OVERRIDE_CONF=0.40, POISON_GUARD_ML_MIN=0.15, SESSION_PIXEL_PERFECT=0.95, VIRTUAL_SEED_BRIGHT_RATIO=0.15, VIRTUAL_SEED_RICH_RATIO=0.15 — kalibrowane na tactical-console / Kentari-launcher (icon_matcher.py:55). VIRTUAL_SEED_* podniesione z 0.07 → 0.15 dnia 2026-07-17 po wizualnym przeglądzie 20 community-mirror crops (`tests/diag_view_community_poison.py`): genuine empty/inactive BOFF slots sięgają ~12% bright/rich, realne mislabeled icons ≥ 19%. Wszelka zmiana wymaga re-kalibracji. | Frozen calibration. |
| Z5-C.5 | **Trainer UI '[empty slot]' / '[inactive slot]' render zostaje.** To jest UX, nie filter. User MUSI widzieć i móc oznaczać empty/inactive — bez tego brak labeling-do-treningu klasy `__empty__`/`__inactive__`. | Nie ruszać. |
| Z5-C.6 | **Layout detector `__boff_*` generation zostaje.** To są internal slot type hints dla Stage 1+, nigdy nie docierają do user view (filter w boff_keys.py:190 i build_writer.py:231 przechwytują). | Nie ruszać. |

---

## 5. Wnioski dla implementacji D-B.3

D-B.3 (komentowanie `sync_client.py:434-436`) jest **bezpieczne** i może
iść w PHASE 4 razem z komentowaniem analogicznego filtra w backendzie
(D-A.1, D-G.1). Bez tego dokumentu nie było wiadomo czy filter w
sync_client chroni USER VIEW czy tylko knowledge.json — okazało się że
**tylko knowledge.json**, i to redundantnie wobec Stage 0 defense-in-depth.

Mechanizm rollback: jeśli po deployu D-A.1 + D-B.3 zaobserwujemy że
Stage 0 defense-in-depth jednak przecieka (np. `_classify_ml` zwraca
real z conf < 0.40 i embedder cross-check nie odrzuca virtual override),
**odkomentowujemy `sync_client.py:434-436` + analogiczne 3 miejsca
w backendzie jednym commitem**. Rollback ma być jednoatomowy.

---

## 6. Out-of-scope

Następujące obszary celowo nie są w tym audycie:

- **Anti-virtual-bias re-tuning** — frozen calibration per Z5-C.4.
- **Recognition pipeline refactor** — Stage 0/1/2/3 architecture niezmienna.
- **BOFF keys algorithm** — `_VIRTUAL_NAMES` filter w `boff_keys.py:190` zostaje.

---

## 7. Log

- 2026-06-09: założenie dokumentu, mapa 8 filtrów USER VIEW + 1 filter INPUT
  (sync_client.py:434-436), 6 decyzji Z5-C.1..Z5-C.6, odblokowanie D-B.3.
