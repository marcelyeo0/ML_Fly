# flybrain — explication complète du projet et guide d'entretien

Ce document explique **tout le projet** (biologie, données, modèles, code, résultats,
limites) et donne de quoi **le présenter et le défendre en entretien**.

---

## 1. Le projet en une phrase

> J'ai simulé le **cerveau entier d'une mouche** (139 000 neurones, 15 millions de
> connexions issues de la microscopie électronique) avec un réseau de neurones à
> impulsions, je l'ai **validé** contre la publication de référence, je l'ai **branché à un
> corps 3D physique** pour qu'il pilote la marche vers une odeur, et j'ai utilisé le même
> graphe pour du **machine learning** (classification de neurones, réseau contraint par
> le connectome).

---

## 2. Le contexte scientifique (à savoir expliquer simplement)

### 2.1 Le connectome FlyWire

- En 2024, le consortium **FlyWire** a publié dans *Nature* la carte complète du cerveau
  d'une drosophile adulte : chaque neurone et chaque synapse, reconstruits à partir
  d'images de microscopie électronique (≈ 7 000 coupes de 40 nm).
- Version utilisée : **v783** → 138 639 neurones, 15,1 M paires (pré, post) connectées,
  54,5 M synapses.
- Chaque connexion a un **nombre de synapses** et un **signe** : le neurotransmetteur
  du neurone présynaptique est prédit par un réseau de neurones à partir de l'image
  (acétylcholine → excitateur, GABA / glutamate → inhibiteur chez la mouche).
- Les neurones sont annotés : *super_class* (sensory, optic, central, descending,
  ascending, motor…), *cell_class*, *cell_type*, côté (gauche/droite).

**Analogie** : le connectome est le « schéma électrique » ; il ne dit pas à quelle vitesse
chaque composant réagit. Simuler, c'est ajouter une physique simple sur ce schéma et
regarder ce qui en émerge.

### 2.2 Le modèle LIF (Leaky Integrate-and-Fire)

Chaque neurone est un circuit RC :

```
tau_m * dv/dt = v_0 - v + g        (potentiel de membrane, fuite vers le repos)
tau_s * dg/dt = -g                 (courant synaptique qui décroît)
si v > v_seuil : spike, v <- v_reset, g <- 0, période réfractaire 2,2 ms
spike présynaptique : g_post += w_syn * n_synapses * signe   (après 1,8 ms de délai)
```

Paramètres de **Shiu et al. 2024 (*Nature*)** : v_0 = −52 mV, seuil −45 mV,
tau_m = 20 ms, tau_s = 5 ms, w_syn = 0,275 mV par synapse. Un seul paramètre libre
(w_syn) pour tout le cerveau. Leur résultat : stimuler les neurones gustatifs du sucre
active le motoneurone **MN9** de la trompe (réflexe d'alimentation), ce qui a été vérifié
expérimentalement.

> Point clé en entretien : *le modèle est volontairement simple ; tout le « savoir » est
> dans le câblage.* C'est ce qui rend le résultat intéressant.

### 2.3 Le corps : NeuroMechFly / flygym

- **NeuroMechFly** (Lobato-Rios 2022, Wang-Chen 2024) : modèle 3D de la mouche
  (87 articulations) dans le moteur physique **MuJoCo**, avec contacts, adhérence des
  pattes, capteurs d'odeur (2 antennes + 2 palpes) et vision (2 × 721 ommatidies).
- **flygym** : la bibliothèque Python qui l'expose comme un environnement type Gym.
- `HybridTurningController` : un générateur central de rythme (CPG) produit la marche
  tripode ; on lui donne **deux amplitudes (gauche, droite)** : ralentir un côté = tourner.

### 2.4 Les projets dont je pars

| projet | ce que c'est | ce que j'en fais |
| --- | --- | --- |
| **NeuroFly** (seven-monarchs) | pipeline cerveau Brian2 + corps flygym + vision flyvis, vidéo split-screen | sous-module ; je le fais tourner sous Windows (runner) et je l'analyse de façon critique |
| **DesktopFly** (DenisSergeevitch) | mouche 3D sur le bureau (Electron + three.js), cerveau 668 neurones FlyWire + 1 045 neurones MaleCNS | sous-module ; installé, testé (tests verts), lancé |
| **Shiu et al. 2024** | modèle LIF Brian2 + résultats de référence | je le **ré-implémente** et je me **valide** contre leurs résultats |

**Soyez transparent en entretien** : NeuroFly et DesktopFly ne sont pas de moi. Ce qui
est à moi : le package `flybrain` (simulateur, validation, boucle fermée pilotée par le
cerveau, analyses de graphe, ML), l'intégration et le diagnostic des limites.

---

## 3. Architecture du dépôt

```
ML_Fly/
├── src/flybrain/          ← mon code (package Python installable, CLI `flybrain`)
│   ├── data.py            chargement du connectome → objet Connectome (matrice creuse)
│   ├── lif.py             simulateur LIF cerveau entier
│   ├── validate.py        comparaison avec la référence Brian2
│   ├── experiments.py     stimulation sensorielle, latéralisation olfactive, ablation
│   ├── graph.py           analyses de graphe
│   ├── classify.py        classification des neurones par leur câblage
│   ├── train.py           RNN contraint par le connectome (PyTorch)
│   ├── closedloop.py      boucle cerveau ↔ corps 3D
│   ├── plots.py, cli.py, compat.py
├── tests/                 tests unitaires (pytest)
├── scripts/run_neurofly.py  lance NeuroFly (durée réduite, correctifs Windows)
├── external/NeuroFly      sous-module (+ données du connectome)
├── external/desktop-fly   sous-module (app Electron)
└── results/               résultats chiffrés cités dans ce document
```

Flux de données :

```
FlyWire (csv/parquet) ──► Connectome ──┬─► LIFNetwork ──► spikes ──► expériences / validation
                                        ├─► graph.py (PageRank, classes, chemins)
                                        ├─► classify.py (features + GBDT)
                                        ├─► train.py (RNN PyTorch)
                                        └─► closedloop.py ◄──► flygym / MuJoCo (corps 3D)
```

---

## 4. Les modules, un par un (quoi, pourquoi, comment)

### 4.1 `data.py` — le connectome comme matrice creuse

- Lit `Completeness_783.csv` (ordre des neurones), `Connectivity_783.parquet`
  (arêtes), `flywire_annotations.tsv` (types).
- Stocke une **liste d'arêtes** (pre, post, poids signé) en `int32/float32` →
  ≈ 180 Mo en RAM au lieu de plusieurs Go pour un DataFrame.
- `matrix()` → matrice `W[post, pre]` **scipy.sparse** (densité 0,08 %, une matrice dense
  ferait 139k² × 4 octets = 77 Go).
- `subnetwork()`, `neighbourhood()` (BFS sur les connexions ≥ n synapses) pour extraire
  des circuits.

### 4.2 `lif.py` — simulateur cerveau entier

**Pourquoi le réécrire ?** NeuroFly utilise Brian2. Sous Windows, le backend C++ de
Brian2 exige Visual Studio (6 Go, absent ici) ; le backend numpy est ~10× plus lent
(l'auteur annonce 2h30–3h pour 10 s *avec* C++). Construire le réseau Brian2 prend déjà
4 à 10 minutes sur ma machine.

**Ce que j'ai fait :**

1. **Intégration exacte** : l'équation est linéaire entre deux spikes, donc on calcule la
   solution analytique sur un pas `dt` :
   `v ← v0 + (v − v0)·e^{−dt/τm} + g·τs/(τs − τm)·(e^{−dt/τs} − e^{−dt/τm})`,
   `g ← g·e^{−dt/τs}`. Pas d'erreur d'Euler (testé contre la formule fermée).
2. **Propagation creuse** : à chaque pas, seuls quelques neurones spikent ; je récupère
   leurs colonnes dans la matrice CSC (gather vectorisé avec `indptr`) et j'accumule avec
   `np.bincount` → coût proportionnel au nombre de spikes, pas à la taille du cerveau.
3. **Délai synaptique** de 1,8 ms par un **buffer circulaire** de 19 pas.
4. **Réfractaire** : on ne garde que la liste des neurones réfractaires (peu nombreux)
   au lieu de masquer 139k valeurs à chaque pas.
5. **Ordre d'exécution identique à Brian2** (mise à jour → seuil → synapses → reset).
6. API **avec état** (`run(25 ms)` appelable en boucle) → indispensable pour la boucle
   fermée où les entrées changent toutes les 25 ms.

**Performance** : ≈ 5 ms par pas de 0,1 ms pour 139k neurones → 1 s de cerveau ≈ 50 s
sur un i5 portable, sans compilateur.

**Validation (`validate.py`)** : je rejoue l'expérience de Shiu et al. (21 neurones
gustatifs « sucre » à 150 Hz, 1 s, connectome v630) et je compare **neurone par neurone**
avec leurs spikes Brian2 publiés (30 essais) :

| métrique | valeur |
| --- | --- |
| corrélation de Pearson des taux | **0,956** |
| corrélation de Spearman | **0,962** |
| recouvrement du top-20 des neurones les plus actifs | 60 % |
| MN9 (motoneurone d'alimentation) | 106 Hz (moi) vs 93 Hz (Brian2) |

Les deux modèles sont stochastiques (entrées de Poisson) : on attend les mêmes
statistiques, pas les mêmes spikes. Mes taux sont ≈ 10–15 % plus élevés en moyenne ; j'ai
vérifié l'ordre d'exécution et la durée réfractaire, l'écart persiste — c'est une limite
documentée (piste : détails d'implémentation de `PoissonInput` / file de spikes Brian2).

### 4.3 `experiments.py` — expériences in silico

- `sensory_activation` : stimule une population, liste les neurones qui répondent.
- `odor_lateralization` : stimule les **récepteurs olfactifs (ORN) gauches** puis
  **droits** et mesure les neurones descendants (DN, la « sortie motrice » du cerveau).
  - Résultat : en moyenne, les DN gauches et droits répondent pareil (≈ 6 Hz) — l'odeur
    est surtout traitée de façon bilatérale.
  - **Mais** certains DN sont latéralisés : **DNa02** gauche répond plus à l'odeur
    gauche (76 vs 61 Hz). DNa02 est justement un neurone de **pilotage** connu
    (Rayshubskiy et al. 2020 : son activité prédit les virages ipsilatéraux). D'autres
    (DNbe007, DNb05) ont une réponse croisée.
  - C'est cohérent avec la biologie, mais l'effet est faible et bruité → d'où la
    sélection statistique (z-score) dans la boucle fermée.
- `ablation` : supprime les sorties d'un groupe de neurones et mesure l'effet.

### 4.4 `closedloop.py` — le cerveau pilote le corps

Toutes les 25 ms :

1. **Capteurs → cerveau** : flygym donne l'intensité d'odeur sur les antennes et palpes
   gauche/droite. Elle fixe le **taux de Poisson** des ~1 100 ORN de chaque côté. La
   vitesse de marche fixe le taux des neurones ascendants (proprioception).
2. **Cerveau** : `LIFNetwork` simule les 139k neurones pendant 25 ms.
3. **Cerveau → corps** : un **readout** des DN donne une commande de virage ; on
   l'envoie au contrôleur de marche (amplitudes gauche/droite).

**Le readout n'est pas entraîné sur la tâche.** Je le calibre en boucle ouverte : je
stimule les ORN gauches puis droits, et je garde les DN dont la différence de spikes est
significative (z de Poisson ≥ 2) avec un poids ±1. Donc *la décision gauche/droite est
calculée par le connectome* ; je choisis seulement quels neurones écouter.

**Différence avec NeuroFly (point fort à mettre en avant)** : en lisant le code de
NeuroFly, le virage vient surtout d'un gradient d'odeur codé à la main :

```python
odor_turn = np.tanh(_lr_asym * 20.0) * ODOR_TURN_K   # ODOR_TURN_K = 2.5
dn_bias   = float(lr_diff_t) * 0.15                  # contribution du cerveau
```

La contribution du cerveau est ~15× plus faible que la règle codée. Dans `flybrain`, le
virage vient **uniquement** des DN. C'est plus honnête scientifiquement… et plus dur :
la navigation est moins « parfaite », mais elle est réellement produite par le modèle.

**Hypothèse assumée** : les antennes sont à 0,5 mm l'une de l'autre, la différence brute
gauche/droite est de quelques %. J'ajoute un **gain de contraste bilatéral** périphérique
(`contrast_gain`) qui représente l'amplification faite par le lobe antennaire chez la
vraie mouche et que le LIF ne capture pas. C'est un paramètre de modélisation, pas une
mesure — à dire clairement.

Résultats de la boucle fermée : voir section 6.

### 4.5 `graph.py` — analyses de graphe

- Degrés entrants/sortants, force synaptique, bilan excitation/inhibition par neurone.
- **PageRank** (itération de puissance sur matrice creuse) : neurones « carrefours ».
- Matrice de connectivité **entre super-classes** (qui parle à qui).
- **Distance en nombre de synapses** (plus court chemin) entre deux populations.
- **Rich club** : densité de connexions entre les 1 % de neurones les plus connectés,
  comparée à la densité globale.

### 4.6 `classify.py` — reconnaître un neurone à son câblage

Question : *peut-on dire si un neurone est sensoriel, optique, central, descendant…
seulement à partir de qui il connecte ?*

- **Features** sans utiliser l'étiquette : degrés, nombre de synapses, bilan E/I, et un
  **embedding spectral** (SVD tronquée de log(1+|W|), 16 composantes pour les entrées
  et 16 pour les sorties : un résumé de « qui parle à qui »).
- **Modèle** : gradient boosting (`HistGradientBoostingClassifier`), split stratifié
  75/25, poids équilibrés par classe.
- **Métrique** : **macro-F1** (moyenne non pondérée des F1 par classe) car les classes
  sont très déséquilibrées (77 539 neurones optiques vs 110 motoneurones) — l'accuracy
  seule serait trompeuse.

| features | accuracy | macro-F1 |
| --- | --- | --- |
| baseline (classe majoritaire) | 0,56 | 0,07 |
| câblage seul | **0,973** | **0,860** |
| + neurotransmetteur | 0,978 | 0,870 |
| + position 3D | 0,981 | 0,897 |

Lecture : les neurones optiques, centraux, sensoriels sont reconnus à > 98 % ; les
**descendants** sont les plus durs (F1 0,70) : peu nombreux et câblés comme des neurones
centraux côté entrées (leurs sorties vont dans la moelle ventrale, absente du connectome
du cerveau).

### 4.7 `train.py` — réseau récurrent contraint par le connectome

Inspiré de **Lappalainen et al. 2024 (*Nature*)** : on garde le **câblage et les signes**
du connectome, on apprend seulement des **gains** par synapse.

```
r(t+1) = (1 − α) r(t) + α · tanh(relu( (S ⊙ M ⊙ softplus(θ)) r(t) + b + B u ))
M = masque du connectome (qui est connecté), S = signe fixé (loi de Dale), θ appris
```

- Tâche : entrée = intensité d'odeur gauche/droite sur les ORN ; sortie = readout
  linéaire des DN qui doit donner `i_gauche − i_droite` (la commande de virage).
- Circuit : voisinage à 2 synapses des ORN (connexions ≥ 30 synapses) + DN qui en
  reçoivent : 4 473 neurones, 286k connexions.
- Stabilité : j'ai d'abord eu une **explosion** (MSE = inf) ; corrigée en normalisant la
  somme des entrées de chaque neurone à 0,9 (matrice contractante à l'init) et en bornant
  l'activation (tanh∘relu).
- **Contrôle** : même entraînement sur un connectome **mélangé** (cibles permutées :
  mêmes degrés et poids, câblage détruit) → mesure si le *câblage spécifique* aide.

Résultats : section 6.

---

## 5. Faire tourner NeuroFly (et les problèmes résolus)

`scripts/run_neurofly.py` exécute le script original sans le modifier (le sous-module
reste propre) en corrigeant à la volée :

| problème | cause | correctif |
| --- | --- | --- |
| 10 s codées en dur (~3 h) | constante `PHYS_DURATION_S` | remplacée en mémoire (`--seconds`) |
| `UnicodeEncodeError` | la console Windows est en cp1252, le script affiche « ● » | sortie reconfigurée en UTF-8 |
| `flybody` non installable | épingle `numpy==1.26.4` incompatible avec `numpy==2.0.2` | non importé par le script → retiré |
| `Brian2==2.9.0` ne s'installe pas | pas de wheel Python 3.10 Windows, il faut MSVC | wheel binaire 2.8 |
| flyvis : `NetworkDir not found` | poids pré-entraînés pas au bon endroit | copiés + `FLYVIS_ROOT_DIR` |
| flyvis : `PermissionError WinError 32` | bug de `datamate` : supprime un fichier HDF5 encore ouvert (OK sous Linux, interdit sous Windows) | `flybrain.compat.patch_datamate_windows()` |
| `retina.py` introuvable | chemin `wenv310/...` codé en dur | redirigé vers le flygym installé |
| Brian2 lent | pas de compilateur C++ | backend numpy (lent) — c'est la motivation de `flybrain.lif` |

---

## 6. Résultats

<!-- RESULTS -->

---

## 7. Limites (à connaître par cœur — c'est ce qui fait la crédibilité)

1. **Modèle neuronal minimal** : pas de canaux ioniques, pas de neuromodulation, pas de
   plasticité, un seul poids par synapse pour tout le cerveau. Les « taux » sont
   qualitatifs.
2. **Le signe vient d'une prédiction** du neurotransmetteur (≈ 90 % de précision), et le
   glutamate est traité comme inhibiteur partout (simplification).
3. **Pas de moelle épinière (VNC)** dans FlyWire : le lien DN → pattes passe par un
   contrôleur de marche (CPG) artificiel, pas par de vrais circuits moteurs. (DesktopFly,
   lui, utilise un extrait de MaleCNS pour ça — piste d'amélioration.)
4. **Encodage sensoriel simplifié** : intensité → taux de Poisson, gain de contraste
   bilatéral choisi à la main.
5. **Calibration du readout** : choisir quels DN écouter est une décision de
   modélisation, même si elle ne regarde pas la tâche de navigation.
6. **Écart de ~10–15 % avec Brian2** sur les taux absolus (corrélation 0,96).
7. **Temps de calcul** : 1 s de cerveau ≈ 1 min sur CPU portable ; les runs longs sont
   coûteux.
8. Le connectome est **un individu** (une femelle) : pas de variabilité inter-individus.

---

## 8. Pitch d'entretien

### 30 secondes

> « J'ai travaillé sur une simulation du cerveau complet de la drosophile à partir du
> connectome FlyWire publié dans *Nature* en 2024 : 139 000 neurones, 15 millions de
> connexions. J'ai réécrit le simulateur de neurones à impulsions pour qu'il tourne sur un
> portable, je l'ai validé contre les résultats publiés avec une corrélation de 0,96, puis
> je l'ai branché à un modèle physique 3D de la mouche : l'odeur active ses récepteurs, le
> cerveau simulé calcule, et ses neurones descendants font tourner le corps. En parallèle,
> j'ai montré qu'on peut reconnaître le type d'un neurone à 97 % seulement à partir de son
> câblage. »

### 2 minutes (structure STAR)

- **Situation** : le connectome complet d'un cerveau existe depuis 2024 ; la question
  est « que peut-on prédire du comportement à partir du câblage seul ? ».
- **Tâche** : reproduire le modèle de référence, le rendre utilisable (rapide, Windows),
  fermer la boucle avec un corps, et exploiter le graphe en ML.
- **Actions** : (1) package Python propre avec tests ; (2) simulateur LIF creux à
  intégration exacte ; (3) validation quantitative contre Brian2 ; (4) analyse critique de
  NeuroFly → le virage y était surtout codé à la main → ma boucle fait piloter le virage
  par les neurones descendants seulement ; (5) classifieur sur embedding spectral ; (6)
  RNN contraint par le connectome avec contrôle par connectome mélangé.
- **Résultats** : r = 0,96 vs référence ; 1 s de cerveau entier en ~1 min sans
  compilateur ; 97 % d'accuracy / 0,86 macro-F1 ; boucle fermée fonctionnelle + vidéo.

### Ce que ça montre de moi

- Lire et **reproduire un papier** (*Nature*) et **valider** quantitativement.
- **Performance** numérique (matrices creuses, vectorisation, complexité).
- **Esprit critique** : j'ai lu le code d'un projet existant et trouvé que son « cerveau »
  pesait peu dans le comportement.
- **ML rigoureux** : baseline, métrique adaptée au déséquilibre, contrôle négatif.
- **Ingénierie** : package, CLI, tests, sous-modules, correctifs Windows, commits
  atomiques.

---

## 9. Questions probables et réponses

**Q : C'est quoi un connectome ? Pourquoi c'est important ?**
R : La liste de tous les neurones et de toutes leurs synapses. C'est le substrat
physique du calcul ; sans lui, on ne peut que deviner les circuits. FlyWire est le premier
cerveau adulte complet d'un animal avec un comportement complexe.

**Q : Pourquoi un modèle LIF et pas un modèle plus réaliste (Hodgkin-Huxley) ?**
R : On ne connaît pas les paramètres biophysiques de 139k neurones ; un modèle riche
aurait des millions de paramètres non contraints. Le LIF a un seul paramètre libre, ce
qui permet de tester l'hypothèse « le câblage suffit à expliquer X ». Et c'est
calculable.

**Q : Comment as-tu validé ton simulateur ?**
R : Trois niveaux : (1) tests unitaires (repos stable, solution analytique exacte,
propagation avec délai, inhibition, période réfractaire, ablation) ; (2) comparaison
neurone par neurone avec les spikes Brian2 publiés : r = 0,96 ; (3) le comportement
attendu (sucre → MN9) est reproduit.

**Q : Pourquoi tes taux sont plus élevés que Brian2 ?**
R : Écart systématique ~10–15 %. J'ai éliminé l'ordre d'exécution et la durée
réfractaire. Restent des détails d'implémentation (file de spikes, `PoissonInput`). La
structure (qui répond, dans quel ordre) est conservée, ce qui compte pour mes usages.

**Q : Comment ton simulateur peut-il être plus rapide que Brian2 ?**
R : Il ne l'est pas face à Brian2 compilé en C++ ; il l'est face à Brian2 sans
compilateur, ma situation sous Windows. Le gain vient de l'intégration exacte, de la
propagation proportionnelle au nombre de spikes et des opérations en place. Pour aller
plus loin : GPU (PyTorch sparse / CuPy) ou mise à jour seulement des neurones actifs.

**Q : Quelle est la complexité d'un pas de simulation ?**
R : O(N) pour la mise à jour des potentiels (vectorisée) + O(Σ degrés sortants des
neurones qui spikent) pour la propagation.

**Q : Pourquoi la macro-F1 et pas l'accuracy ?**
R : 56 % des neurones sont optiques : prédire « optique » partout donne 56 % d'accuracy
et 0,07 de macro-F1. La macro-F1 donne le même poids à chaque classe.

**Q : Il n'y a pas de fuite de données dans ton classifieur ?**
R : Les features n'utilisent pas l'étiquette (seulement la structure du graphe et,
optionnellement, le neurotransmetteur/la position). L'embedding SVD est non supervisé.
Limite : il est calculé sur tout le graphe (transductif), ce qui est standard en
classification de nœuds, mais un test inductif (sous-graphe non vu) serait plus strict.

**Q : C'est quoi un réseau contraint par le connectome ? Pourquoi c'est utile ?**
R : Un RNN dont la matrice de poids a la structure exacte du connectome (zéros imposés,
signes fixés). On apprend seulement des gains. Ça réduit énormément l'espace des
solutions et permet de prédire l'activité de neurones jamais enregistrés (Lappalainen
2024 l'a montré pour le système visuel).

**Q : Pourquoi comparer à un connectome mélangé ?**
R : C'est le contrôle négatif : même nombre de connexions, mêmes degrés, mêmes poids,
mais câblage aléatoire. Si le vrai connectome fait mieux, c'est que le *câblage
spécifique* porte de l'information, pas juste les statistiques du graphe.

**Q : Dans ta boucle fermée, qu'est-ce qui est vraiment produit par le cerveau ?**
R : La commande de virage, entièrement. Les choix de modélisation sont : l'encodage
odeur → taux de Poisson (avec gain de contraste), le choix des DN lus (calibré en boucle
ouverte, sans regarder la navigation), et le générateur de marche (CPG) qui remplace la
moelle épinière absente du connectome.

**Q : Qu'as-tu appris de NeuroFly ?**
R : C'est une belle démo, mais en lisant le code, le virage vient d'une règle codée
(`2,5 × tanh(20 × asymétrie)`) et le cerveau ne contribue qu'à `0,15 × asymétrie DN`.
J'ai voulu une boucle où le comportement dépend vraiment du modèle neuronal.

**Q : Quelles sont les limites principales ?**
R : Voir section 7 — modèle neuronal minimal, signes prédits, pas de moelle ventrale,
encodage sensoriel choisi à la main, coût de calcul.

**Q : Si tu avais plus de temps ?**
R : (1) GPU pour simuler 10 s en temps raisonnable ; (2) brancher le circuit locomoteur
MaleCNS (moelle ventrale) entre DN et pattes, comme DesktopFly ; (3) entraîner le RNN
contraint sur des données d'imagerie réelles ; (4) évaluer la navigation sur de nombreux
essais avec statistiques (taux de succès, temps d'arrivée) et contrôles (readout
aléatoire, connectome mélangé).

**Q : Comment as-tu organisé le code ?**
R : Package installable (`pyproject.toml`), modules séparés par responsabilité, CLI
unique, tests pytest, ruff, sous-modules git pour le code tiers, résultats reproductibles
par commande, commits atomiques en Conventional Commits.

---

## 10. Glossaire express

| terme | définition |
| --- | --- |
| spike / potentiel d'action | impulsion électrique émise quand le neurone dépasse son seuil |
| LIF | modèle « intègre et fuit » : le potentiel intègre les entrées et fuit vers le repos |
| Poisson (entrée) | spikes aléatoires indépendants à un taux donné |
| ORN | neurone récepteur olfactif (antennes, palpes) |
| GRN | neurone récepteur gustatif |
| DN | neurone descendant : du cerveau vers la moelle ventrale, commande motrice |
| ascendant | de la moelle vers le cerveau (proprioception) |
| MN9 | motoneurone qui contrôle la trompe (alimentation) |
| SEZ | zone sous-œsophagienne : goût, alimentation |
| VNC | moelle ventrale de l'insecte (pattes, ailes) |
| CPG | générateur central de rythme (oscillateurs de marche) |
| loi de Dale | un neurone libère le même neurotransmetteur à toutes ses synapses |
| macro-F1 | moyenne des F1 de chaque classe, sans pondération par effectif |
| SVD tronquée | décomposition en valeurs singulières, garde les k plus grandes |
| PageRank | importance d'un nœud selon l'importance de ceux qui pointent vers lui |
| MuJoCo | moteur de simulation physique (corps rigides, contacts) |

---

## 11. Reproduire

```bash
flybrain validate --trials 10      # ~10 min
flybrain classify                  # ~1,5 min
flybrain graph                     # ~1 min
flybrain train --epochs 100        # long sur CPU
flybrain closedloop --seconds 2    # long : ~1 min de calcul par 0,1 s simulée + calibration
flybrain desktop                   # mouche sur le bureau
```

Les sorties lourdes vont dans `outputs/` (ignoré par git), les résultats cités dans
`results/`.
